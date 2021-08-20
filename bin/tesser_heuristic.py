"""Heuristic file for use with heudiconv."""


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
    # find the functional runs that were completed
    include_sbref = []
    include_bold = []
    for i, s in enumerate(seqinfo):
        if s.series_description == 'functional_run':
            # should be 298 TRs (one participant shows as 596 for some reason)
            if s.dim4 in [298, 596]:
                # include this scan and the one before, which is the SBRef
                include_sbref.append(seqinfo[i - 1].series_id)
                include_bold.append(s.series_id)

    # define file path format for BIDS
    key_T1w = create_key('sub-{subject}/anat/sub-{subject}_T1w')
    key_T2w = create_key('sub-{subject}/anat/sub-{subject}_run-{item}_T2w')
    key_magnitude = create_key('sub-{subject}/fmap/sub-{subject}_run-{item}_magnitude')
    key_phasediff = create_key('sub-{subject}/fmap/sub-{subject}_run-{item}_phasediff')
    key_struct = create_key(
        'sub-{subject}/func/sub-{subject}_task-struct_run-{item}_bold'
    )
    key_sbref = create_key(
        'sub-{subject}/func/sub-{subject}_task-struct_run-{item}_sbref'
    )

    # sort scans into file types
    info = {
        key_T1w: [],
        key_T2w: [],
        key_magnitude: [],
        key_phasediff: [],
        key_sbref: [],
        key_struct: [],
    }
    n_fieldmap = 0
    n_T2 = 0
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
            # T1 highres anatomical
            info[key_T1w].append(s.series_id)

        elif 'T2' in s.series_description:
            # T2 coronal anatomical
            n_T2 += 1

            # for participants with extra scans, exclude the worst ones
            if s.patient_id == 'tesser_101' and n_T2 == 1:
                continue
            elif s.patient_id == 'TESSER_108' and n_T2 == 3:
                continue

            # add scan if included
            info[key_T2w].append(s.series_id)

        elif s.series_description == 'fieldmap':
            # fieldmaps to estimate susceptibility distortion
            n_fieldmap += 1

            # if extra fieldmaps, exclude the earlier ones
            if s.patient_id == 'tesser_105' and n_fieldmap in [1, 2]:
                continue
            elif s.patient_id == 'TESSER_109' and n_fieldmap in [3, 4]:
                continue
            elif s.patient_id == 'tesser_122' and n_fieldmap in [1, 2]:
                continue

            # add to either magnitude or phase difference image set
            if n_fieldmap % 2 == 1:
                info[key_magnitude].append(s.series_id)
            else:
                info[key_phasediff].append(s.series_id)

        elif s.series_description == 'functional_run_SBRef':
            # single-band image from a functional scan
            if s.series_id in include_sbref:
                info[key_sbref].append(s.series_id)

        elif s.series_description == 'functional_run':
            # functional scan
            if s.series_id in include_bold:
                info[key_struct].append(s.series_id)
    return info
