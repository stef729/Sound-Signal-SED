#
# Implements the core metrics from sound event detection evaluation module http://tut-arg.github.io/sed_eval/ and
# The DOA metrics are explained in the SELDnet paper
#

import numpy as np
from scipy.optimize import linear_sum_assignment
from IPython import embed
eps = np.finfo(np.float).eps


##########################################################################################
# SELD scoring functions - class implementation
#
# NOTE: Supports only one-hot labels for both SED and DOA. Doesnt work for baseline method
# directly, since it estimated DOA in regression approach. Check below the class for
# one shot (function) implementations of all metrics. The function implementation has
# support for both one-hot labels and regression values of DOA estimation.
##########################################################################################

class SELDMetrics(object):
    def __init__(self, nb_frames_1s=None, data_gen=None):
        # SED params
        self._S = 0
        self._D = 0
        self._I = 0
        self._TP = 0
        self._Nref = 0
        self._Nsys = 0
        self._block_size = nb_frames_1s

        # DOA params
        self._doa_loss_pred_cnt = 0
        self._nb_frames = 0

        self._doa_loss_pred = 0
        self._nb_good_pks = 0

        self._data_gen = data_gen

        self._less_est_cnt, self._less_est_frame_cnt = 0, 0
        self._more_est_cnt, self._more_est_frame_cnt = 0, 0

    def f1_overall_framewise(self, O, T):
        TP = ((2 * T - O) == 1).sum()
        Nref, Nsys = T.sum(), O.sum()
        self._TP += TP
        self._Nref += Nref
        self._Nsys += Nsys

    def er_overall_framewise(self, O, T):
        FP = np.logical_and(T == 0, O == 1).sum(1)
        FN = np.logical_and(T == 1, O == 0).sum(1)
        S = np.minimum(FP, FN).sum()
        D = np.maximum(0, FN - FP).sum()
        I = np.maximum(0, FP - FN).sum()
        self._S += S
        self._D += D
        self._I += I

    def f1_overall_1sec(self, O, T):
        new_size = int(np.ceil(O.shape[0] / self._block_size))
        O_block = np.zeros((new_size, O.shape[1]))
        T_block = np.zeros((new_size, O.shape[1]))
        for i in range(0, new_size):
            O_block[i, :] = np.max(O[int(i * self._block_size):int(i * self._block_size + self._block_size - 1), :], axis=0)
            T_block[i, :] = np.max(T[int(i * self._block_size):int(i * self._block_size + self._block_size - 1), :], axis=0)
        return self.f1_overall_framewise(O_block, T_block)

    def er_overall_1sec(self, O, T):
        new_size = int(O.shape[0] / self._block_size)
        O_block = np.zeros((new_size, O.shape[1]))
        T_block = np.zeros((new_size, O.shape[1]))
        for i in range(0, new_size):
            O_block[i, :] = np.max(O[int(i * self._block_size):int(i * self._block_size + self._block_size - 1), :], axis=0)
            T_block[i, :] = np.max(T[int(i * self._block_size):int(i * self._block_size + self._block_size - 1), :], axis=0)
        return self.er_overall_framewise(O_block, T_block)

    def update_sed_scores(self, pred, gt):
        """
        Computes SED metrics for one second segments

        :param pred: predicted matrix of dimension [nb_frames, nb_classes], with 1 when sound event is active else 0
        :param gt:  reference matrix of dimension [nb_frames, nb_classes], with 1 when sound event is active else 0
        :param nb_frames_1s: integer, number of frames in one second
        :return:
        """
        self.f1_overall_1sec(pred, gt)
        self.er_overall_1sec(pred, gt)

    def compute_sed_scores(self):
        ER = (self._S + self._D + self._I) / (self._Nref + 0.0)

        prec = float(self._TP) / float(self._Nsys + eps)
        recall = float(self._TP) / float(self._Nref + eps)
        F = 2 * prec * recall / (prec + recall + eps)

        return ER, F


    def reset(self):
        # SED params
        self._S = 0
        self._D = 0
        self._I = 0
        self._TP = 0
        self._Nref = 0
        self._Nsys = 0

        # DOA params
        self._doa_loss_pred_cnt = 0
        self._nb_frames = 0

        self._doa_loss_pred = 0
        self._nb_good_pks = 0

        self._less_est_cnt, self._less_est_frame_cnt = 0, 0
        self._more_est_cnt, self._more_est_frame_cnt = 0, 0


###############################################################
# SED scoring functions
###############################################################


def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])


def f1_overall_framewise(O, T):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    TP = ((2 * T - O) == 1).sum()
    Nref, Nsys = T.sum(), O.sum()

    prec = float(TP) / float(Nsys + eps)
    recall = float(TP) / float(Nref + eps)
    f1_score = 2 * prec * recall / (prec + recall + eps)
    return f1_score


def er_overall_framewise(O, T):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)

    FP = np.logical_and(T == 0, O == 1).sum(1)
    FN = np.logical_and(T == 1, O == 0).sum(1)

    S = np.minimum(FP, FN).sum()
    D = np.maximum(0, FN-FP).sum()
    I = np.maximum(0, FP-FN).sum()

    Nref = T.sum()
    ER = (S+D+I) / (Nref + 0.0)
    return ER


def f1_overall_1sec(O, T, block_size):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    new_size = int(np.ceil(O.shape[0] / block_size))
    O_block = np.zeros((new_size, O.shape[1]))
    T_block = np.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block[i, :] = np.max(O[int(i * block_size):int(i * block_size + block_size - 1), :], axis=0)
        T_block[i, :] = np.max(T[int(i * block_size):int(i * block_size + block_size - 1), :], axis=0)
    return f1_overall_framewise(O_block, T_block)


def er_overall_1sec(O, T, block_size):
    if len(O.shape) == 3:
        O, T = reshape_3Dto2D(O), reshape_3Dto2D(T)
    new_size = int(O.shape[0] / (block_size))
    O_block = np.zeros((new_size, O.shape[1]))
    T_block = np.zeros((new_size, O.shape[1]))
    for i in range(0, new_size):
        O_block[i, :] = np.max(O[int(i * block_size):int(i * block_size + block_size - 1), :], axis=0)
        T_block[i, :] = np.max(T[int(i * block_size):int(i * block_size + block_size - 1), :], axis=0)
    return er_overall_framewise(O_block, T_block)


def compute_sed_scores(pred, gt, nb_frames_1s):
    """
    Computes SED metrics for one second segments

    :param pred: predicted matrix of dimension [nb_frames, nb_classes], with 1 when sound event is active else 0
    :param gt:  reference matrix of dimension [nb_frames, nb_classes], with 1 when sound event is active else 0
    :param nb_frames_1s: integer, number of frames in one second
    :return:
    """
    f1o = f1_overall_1sec(pred, gt, nb_frames_1s)
    ero = er_overall_1sec(pred, gt, nb_frames_1s)
    scores = [ero, f1o]
    return scores




###############################################################
# SELD scoring functions
###############################################################


def compute_seld_metric(sed_error):
    """
    Compute SELD metric from sed and doa errors.

    :param sed_error: [error rate (0 to 1 range), f score (0 to 1 range)]
    :param doa_error: [doa error (in degrees), frame recall (0 to 1 range)]
    :return: seld metric result
    """
    seld_metric = np.mean([
        sed_error[0],
        1 - sed_error[1]]
        )
    return seld_metric


def compute_seld_metrics_from_output_format_dict(_pred_dict, _gt_dict, _feat_cls):
    """
        Compute SELD metrics between _gt_dict and_pred_dict in DCASE output format

    :param _pred_dict: dcase output format dict
    :param _gt_dict: dcase output format dict
    :param _feat_cls: feature or data generator class
    :return: the seld metrics
    """
    _gt_labels = output_format_dict_to_classification_labels(_gt_dict, _feat_cls)
    _pred_labels = output_format_dict_to_classification_labels(_pred_dict, _feat_cls)
    _er, _f = compute_sed_scores(_pred_labels.max(2), _gt_labels.max(2), _feat_cls.nb_frames_1s())
    return  _er, _f


###############################################################
# Functions for format conversions
###############################################################

def output_format_dict_to_classification_labels(_output_dict, _feat_cls):

    _unique_classes = _feat_cls.get_classes()
    _nb_classes = len(_unique_classes)
    _max_frames = _feat_cls.get_nb_frames()
    _labels = np.zeros((_max_frames, _nb_classes))

    for _frame_cnt in _output_dict.keys():
        if _frame_cnt < _max_frames:
            for _tmp_doa in _output_dict[_frame_cnt]:                
                # create label
                _labels[_frame_cnt, _tmp_doa[0]] = 1

    return _labels


def regression_label_format_to_output_format(_feat_cls, _sed_labels, _doa_labels_deg):
    """
    Converts the sed (classification) and doa labels predicted in regression format to dcase output format.

    :param _feat_cls: feature or data generator class instance
    :param _sed_labels: SED labels matrix [nb_frames, nb_classes]
    :param _doa_labels_deg: DOA labels matrix [nb_frames, 2*nb_classes] in degrees
    :return: _output_dict: returns a dict containing dcase output format
    """

    _unique_classes = _feat_cls.get_classes()
    _nb_classes = len(_unique_classes)
    _azi_labels = _doa_labels_deg[:, :_nb_classes]
    _ele_labels = _doa_labels_deg[:, _nb_classes:]

    _output_dict = {}
    for _frame_ind in range(_sed_labels.shape[0]):
        _tmp_ind = np.where(_sed_labels[_frame_ind, :])
        if len(_tmp_ind[0]):
            _output_dict[_frame_ind] = []
            for _tmp_class in _tmp_ind[0]:
                _output_dict[_frame_ind].append([_tmp_class, _azi_labels[_frame_ind, _tmp_class], _ele_labels[_frame_ind, _tmp_class]])
    return _output_dict


def classification_label_format_to_output_format(_feat_cls, _labels):
    """
    Converts the seld labels predicted in classification format to dcase output format.

    :param _feat_cls: feature or data generator class instance
    :param _labels: SED labels matrix [nb_frames, nb_classes, nb_azi*nb_ele]
    :return: _output_dict: returns a dict containing dcase output format
    """
    _output_dict = {}
    for _frame_ind in range(_labels.shape[0]):
        _tmp_class_ind = np.where(_labels[_frame_ind].sum(1))
        if len(_tmp_class_ind[0]):
            _output_dict[_frame_ind] = []
            for _tmp_class in _tmp_class_ind[0]:
                _tmp_spatial_ind = np.where(_labels[_frame_ind, _tmp_class])
                for _tmp_spatial in _tmp_spatial_ind[0]:
                    _azi, _ele = _feat_cls.get_matrix_index(_tmp_spatial)
                    _output_dict[_frame_ind].append(
                        [_tmp_class, _azi, _ele])

    return _output_dict


def description_file_to_output_format(_desc_file_dict, _unique_classes, _hop_length_sec):
    """
    Reads description file in csv format. Outputs, the dcase format results in dictionary, and additionally writes it
    to the _output_file

    :param _unique_classes: unique classes dictionary, maps class name to class index
    :param _desc_file_dict: full path of the description file
    :param _hop_length_sec: hop length in seconds

    :return: _output_dict: dcase output in dicitionary format
    """

    _output_dict = {}
    for _ind, _tmp_start_sec in enumerate(_desc_file_dict['start']):
        _tmp_class = _unique_classes[_desc_file_dict['class'][_ind]]
        _tmp_azi = _desc_file_dict['azi'][_ind]
        _tmp_ele = _desc_file_dict['ele'][_ind]
        _tmp_end_sec = _desc_file_dict['end'][_ind]

        _start_frame = int(_tmp_start_sec / _hop_length_sec)
        _end_frame = int(_tmp_end_sec / _hop_length_sec)
        for _frame_ind in range(_start_frame, _end_frame + 1):
            if _frame_ind not in _output_dict:
                _output_dict[_frame_ind] = []
            _output_dict[_frame_ind].append([_tmp_class, _tmp_azi, _tmp_ele])

    return _output_dict


def load_output_format_file(_output_format_file):
    """
    Loads DCASE output format csv file and returns it in dictionary format

    :param _output_format_file: DCASE output format CSV
    :return: _output_dict: dictionary
    """
    _output_dict = {}
    _fid = open(_output_format_file, 'r')
    # next(_fid)
    for _line in _fid:
        _words = _line.strip().split(',')
        _frame_ind = int(_words[0])
        if _frame_ind not in _output_dict:
            _output_dict[_frame_ind] = []
        _output_dict[_frame_ind].append([int(_words[1])])
    _fid.close()
    return _output_dict


def write_output_format_file(_output_format_file, _output_format_dict):
    """
    Writes DCASE output format csv file, given output format dictionary

    :param _output_format_file:
    :param _output_format_dict:
    :return:
    """
    _fid = open(_output_format_file, 'w')
    # _fid.write('{},{},{},{}\n'.format('frame number with 20ms hop (int)', 'class index (int)', 'azimuth angle (int)', 'elevation angle (int)'))
    for _frame_ind in _output_format_dict.keys():
        for _value in _output_format_dict[_frame_ind]:
            _fid.write('{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), int(_value[1]), int(_value[2])))
    _fid.close()
