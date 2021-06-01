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

    key_T1w = create_key('sub-{subject}/anat/sub-{subject}_T1w')
    key_T2w = create_key('sub-{subject}/anat/sub-{subject}_run-{item}_T2w')
    key_magnitude = create_key('sub-{subject}/fmap/sub-{subject}_run-{item}_magnitude')
    key_phasediff = create_key('sub-{subject}/fmap/sub-{subject}_run-{item}_phasediff')
    key_struct = create_key('sub-{subject}/func/sub-{subject}_task-struct_run-{item}_bold')
    key_sbref = create_key('sub-{subject}/func/sub-{subject}_task-struct_run-{item}_sbref')

    info = {
        key_T1w: [],
        key_T2w: [],
        key_magnitude: [],
        key_phasediff: [],
        key_sbref: [],
        key_struct: [],
    }

    # get just the functional runs that were completed
    include_sbref = []
    include_bold = []
    for i, s in enumerate(seqinfo):
        if s.series_description == 'functional_run':
            if s.dim4 == 298:
                # include this scan and the one before, which is the SBRef
                include_sbref.append(seqinfo[i - 1].series_id)
                include_bold.append(s.series_id)

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
        if s.series_description == 'mprage':
            info[key_T1w].append(s.series_id)
        elif 'T2' in s.series_description:
            info[key_T2w].append(s.series_id)
        elif s.series_description == 'fieldmap':
            n_fieldmap += 1
            if n_fieldmap % 2 == 1:
                info[key_magnitude].append(s.series_id)
            else:
                info[key_phasediff].append(s.series_id)
        elif s.series_description == 'functional_run_SBRef':
            if s.series_id in include_sbref:
                info[key_sbref].append(s.series_id)
        elif s.series_description == 'functional_run':
            if s.series_id in include_bold:
                info[key_struct].append(s.series_id)
    return info
