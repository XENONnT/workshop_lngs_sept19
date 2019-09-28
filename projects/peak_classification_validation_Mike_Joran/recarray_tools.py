"""Utility to match peaks from results of different processor versions / processor and simulator
Jelle Aalbers, Nikhef, September 2015
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from tqdm import tqdm

import recarray_tools as rt
from multihist import Hist1d
try:
    from pax.datastructure import INT_NAN
except ImportError:
    INT_NAN = -99999


def match_peaks(allpeaks1, allpeaks2, matching_fuzz=1, unknown_types=(b'unknown', b'lone_hit', b'coincidence'),
                keep_tpc_only=True):
    """Perform peak matching between two numpy record arrays with fields: Event, left, right, type, area
    If a peak is split into many fragments (e.g. two close peaks split into three peaks),
    the results are unreliable and depend on which peak set is peaks1 and which is peaks2.

    Returns (allpeaks1, allpeaks2), each with three extra fields: id, outcome, matched_to
        id: unique number for each peak
        outcome: Can be one of:
            found:  Peak was matched 1-1 between peaks1 and peaks2 (type agrees, no other peaks in range).
                    Note that area, widths, etc. can still be quite different!
            missed: Peak is not present in the other list
            misid_as_XX: Peak is present in the other list, but has type XX
            merged: Peak is merged with another peak in the other list, the new 'super-peak' has the same type
            merged_to_XX: As above, but 'super-peak' has type XX
            split: Peak is split in the other list, but more than one fragment has the same type as the parent.
            chopped: As split, but one or several fragments are unclassified, exactly one has the correct type.
            split_and_unclassified: As split, but all fragments are unclassified in the other list.
            split_and_misid: As split, but at least one fragment has a different peak type.
        matched_to: id of matching in peak in the other list if outcome is found or misid_as_XX, INT_NAN otherwise.
    """

    # Keep only tpc_peaks
    if keep_tpc_only:
        if 'detector' in allpeaks1.dtype.names:
            allpeaks1 = allpeaks1[allpeaks1['detector'] == b'tpc']
        if 'detector' in allpeaks2.dtype.names:
            allpeaks2 = allpeaks2[allpeaks2['detector'] == b'tpc']

    # Remove true photoionization afterpulse peaks (they were not in initial instruction file)
    allpeaks1 = allpeaks1[allpeaks1['type'] != b'photoionization_afterpulse']
    allpeaks2 = allpeaks2[allpeaks2['type'] != b'photoionization_afterpulse']

    # Append id, outcome and matched_to fields
    print("\tAppending extra fields...\n")
    allpeaks1 = rt.append_fields(allpeaks1,
                                 ('id', 'outcome', 'matched_to'),
                                 (np.arange(len(allpeaks1)),
                                  np.array(['missed'] * len(allpeaks1), dtype='S32'),
                                  INT_NAN * np.ones(len(allpeaks1), dtype=np.int64)))
    allpeaks2 = rt.append_fields(allpeaks2,
                                 ('id', 'outcome', 'matched_to'),
                                 (np.arange(len(allpeaks2)),
                                 np.array(['missed'] * len(allpeaks2), dtype='S32'),
                                 INT_NAN * np.ones(len(allpeaks2), dtype=np.int64)))

    # Group each peak by event in OrderedDict
    print("\tGrouping peaks 1 by event...\n")
    peaks1_by_event = rt.dict_group_by(allpeaks1, 'Event')
    print("\tGrouping peaks 2 by event...\n")
    peaks2_by_event = rt.dict_group_by(allpeaks2, 'Event')

    for event, peaks_1 in tqdm(peaks1_by_event.items(), desc='Matching peaks'):
        if event not in peaks2_by_event:
            continue
        peaks_2 = peaks2_by_event[event]

        for p1_i, p1 in enumerate(peaks_1):
            # Select all found peaks that overlap at least partially with the true peak
            selection = (peaks_2['left'] <= p1['right'] + matching_fuzz) & \
                        (peaks_2['right'] >= p1['left'] - matching_fuzz)
            matching_peaks = peaks_2[selection]

            if len(matching_peaks) == 0:
                # Peak was missed; that's the default outcome, no need to set anything
                pass

            elif len(matching_peaks) == 1:
                # A unique match! Hurray!
                p2 = matching_peaks[0]
                p1['matched_to'] = p2['id']
                p2['matched_to'] = p1['id']
                # Do the types match?
                if p1['type'] == p2['type']:
                    p1['outcome'] = p2['outcome'] = 'found'
                else:
                    if p1['type'] in unknown_types:
                        p2['outcome'] = 'unclassified'
                    else:
                        p2['outcome'] = 'misid_as_%s' % p1['type'].decode()
                    if p2['type'] in unknown_types:
                        p1['outcome'] = 'unclassified'
                    else:
                        p1['outcome'] = 'misid_as_%s' % p2['type'].decode()
                    # If the peaks are unknown in both sets, they will count as 'found'.
                    # Hmm....
                matching_peaks[0] = p2
            else:
                # More than one peak overlaps p1
                handle_peak_merge(parent=p1, fragments=matching_peaks, unknown_types=unknown_types)
                
            # matching_peaks is a copy, not a view, so we have to copy the results over to peaks_2 manually
            # Sometimes I wish python had references...
            for i_in_matching_peaks, i_in_peaks_2 in enumerate(np.where(selection)[0]):
                peaks_2[i_in_peaks_2] = matching_peaks[i_in_matching_peaks]

        # Match in reverse to detect merged peaks
        # >1 peaks in 1 may claim to be matched to a peak in 2, in which case we should correct the outcome...
        for p2_i, p2 in enumerate(peaks_2):
            selection = peaks_1['matched_to'] == p2['id']
            matching_peaks = peaks_1[selection]
            if len(matching_peaks) > 1:
                handle_peak_merge(parent=p2, fragments=matching_peaks, unknown_types=unknown_types)
                
            # matching_peaks is a copy, not a view, so we have to copy the results over to peaks_1 manually
            # Sometimes I wish python had references...
            for i_in_matching_peaks, i_in_peaks_1 in enumerate(np.where(selection)[0]):
                peaks_1[i_in_peaks_1] = matching_peaks[i_in_matching_peaks]
                    


    # Concatenate peaks again into result list
    # Necessary because group_by (and np.split inside that) returns copies, not views
    return np.concatenate(list(peaks1_by_event.values())), \
           np.concatenate(list(peaks2_by_event.values()))


def handle_peak_merge(parent, fragments, unknown_types):
    found_types = fragments['type']
    is_ok = found_types == parent['type']
    is_unknown = np.in1d(found_types, unknown_types)
    is_misclass = (True ^ is_ok) & (True ^ is_unknown)
    # We have to loop over the fragments to avoid making a copy
    for i in range(len(fragments)):
        if is_unknown[i] or is_misclass[i]:
            if parent['type'] in unknown_types:
                fragments[i]['outcome'] = 'merged_to_unknown'
            else:
                fragments[i]['outcome'] = 'merged_to_%s' % parent['type'].decode()
        else:
            fragments[i]['outcome'] = 'merged'
        # Link the fragments to the parent
        fragments[i]['matched_to'] = parent['id']
    if np.any(is_misclass):
        parent['outcome'] = 'split_and_misid'
    # All fragments are either ok or unknown
    # If more than one fragment is given the same class
    # as the parent peak, then call it "split"
    elif len(np.where(is_ok)[0]) > 1:
        parent['outcome'] = 'split'
    elif np.all(is_unknown):
        parent['outcome'] = 'split_and_unclassified'
    # If exactly one fragment out of > 1 fragments
    # is correctly classified, then call the parent chopped
    else:
        parent['outcome'] = 'chopped'
    # We can't link the parent to all fragments... link to the largest one:
    parent['matched_to'] = fragments[np.argmax(fragments['area'])]['id']


outcome_colors = {
    'found':            'darkblue',
    'chopped':          'mediumslateblue',

    'missed':           'red',
    'merged':           'turquoise',
    'split':            'purple',

    'misid_as_s2':      'orange',
    'misid_as_s1':      'goldenrod',
    'split_and_misid':  'darkorange',
    'merged_to_s2':     'chocolate',
    'merged_to_s1':     'sandybrown',
    'merged_to_unknown': 'khaki',

    'unclassified':     'green',
    'split_and_unclassified':     'seagreen',
    'merged_and_unclassified':    'limegreen',
}



def peak_matching_histogram(results, histogram_key, bins=10):
    """Make 1D histogram of peak matching results (=peaks with extra fields added by matagainst histogram_key"""

    if histogram_key not in results.dtype.names:
        raise ValueError('Histogram key %s should be one of the columns in results: %s' % (histogram_key,
                                                                                           results.dtype.names))

    # How many true peaks do we have in each bin in total?
    n_peaks_hist = Hist1d(results[histogram_key], bins)
    hists = {'_total': n_peaks_hist}

    for outcome in np.unique(results['outcome']):
        # Histogram the # of peaks that have this outcome
        hist = Hist1d(results[results['outcome'] == outcome][histogram_key],
                      bins=n_peaks_hist.bin_edges)
        outcome = outcome.decode()
        hists[outcome] = hist

    return hists


def plot_peak_matching_histogram(*args, **kwargs):
    hists = peak_matching_histogram(*args, **kwargs)
    _plot_peak_matching_histogram(hists)


def _plot_peak_matching_histogram(hists):
    """Make 1D histogram of peak matching results (=peaks with extra fields added by matagainst histogram_key"""

    n_peaks_hist = hists['_total']

    for outcome, hist in hists.items():
        hist = hist.histogram.astype(np.float)

        if outcome == '_total':
            continue

        print("\t%0.2f%% %s" % (100 * hist.sum()/n_peaks_hist.n, outcome))

        # Compute Errors on estimate of a proportion
        # Should have vectorized this... lazy
        # Man this code is ugly!!!!
        limits_d = []
        limits_u = []
        for i, x in enumerate(hist):
            limit_d, limit_u = binom_interval(x, total=n_peaks_hist.histogram[i])
            limits_d.append(limit_d)
            limits_u.append(limit_u)
        limits_d = np.array(limits_d)
        limits_u = np.array(limits_u)

        # Convert hist to proportion
        hist /= n_peaks_hist.histogram.astype('float')

        color = outcome_colors.get(outcome, np.random.rand(3,))
        plt.errorbar(x=n_peaks_hist.bin_centers,
                     y=hist,
                     yerr=[hist - limits_d, limits_u - hist],
                     label=outcome,
                     color=color,
                     linestyle='-' if outcome == 'found' else '',
                     marker='s')

        # Wald intervals: not so good
        # errors = np.sqrt(
        #     hist*(1-hist)/all_true_peaks_histogram
        # )
        # plt.errorbar(x=bin_centers, y=hist, yerr = errors, label=outcome)

    plt.xlim(n_peaks_hist.bin_edges[0], n_peaks_hist.bin_edges[-1])
    plt.ylabel('Fraction of peaks')
    plt.ylim(0, 1)
    plt.legend(loc='lower right', shadow=True)
    legend = plt.legend(loc='best', prop={'size': 10})
    if legend and legend.get_frame():
        legend.get_frame().set_alpha(0.8)


def binom_interval(success, total, conf_level=0.95):
    """Confidence interval on binomial - using Jeffreys interval
    Code stolen from https://gist.github.com/paulgb/6627336
    Agrees with http://statpages.info/confint.html for binom_interval(1, 10)
    """
    # TODO: special case for success = 0 or = total? see wikipedia
    quantile = (1 - conf_level) / 2.
    lower = beta.ppf(quantile, success, total - success + 1)
    upper = beta.ppf(1 - quantile, success + 1, total - success)
    # If something went wrong with a limit calculation, report the trivial limit
    if np.isnan(lower):
        lower = 0
    if np.isnan(upper):
        upper = 1
    return lower, upper
