import os


def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes


def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where

    allowed template fields - follow python string module:

    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    """

    key_magnitude = create_key('sub-{subject}/fmap/sub-{subject}_run-{item}_magnitude1')
    key_phasediff = create_key('sub-{subject}/fmap/sub-{subject}_run-{item}-phasediff')
    info = {key_magnitude: [], key_phasediff: []}

    n_fieldmap = 0
    for s in seqinfo:
        """
        The namedtuple `s` contains the following fields:

        * total_files_till_now
        * example_dcm_file
        * series_id
        * dcm_dir_name
        * unspecified2
        * unspecified3
        * dim1
        * dim2
        * dim3
        * dim4
        * TR
        * TE
        * protocol_name
        * is_motion_corrected
        * is_derived
        * patient_id
        * study_description
        * referring_physician_name
        * series_description
        * image_type
        """
        if s.series_description == 'fieldmap':
            n_fieldmap += 1
            if n_fieldmap % 2 == 1:
                info[key_magnitude].append(s.series_id)
            else:
                info[key_phasediff].append(s.series_id)
    return info
