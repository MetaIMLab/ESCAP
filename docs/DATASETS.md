# Dataset Description

ESCAP evaluates multi-source cardiac CT segmentation under heterogeneous
annotation protocols. The following descriptions summarize the three source
datasets. No dataset files are included here.

## CHD68

CHD68 contains 68 cardiac CT volumes from 14 congenital heart disease types.
The seven annotated structures are myocardium, left atrium, left ventricle,
right atrium, right ventricle, ascending aorta, and pulmonary artery. The
protocol includes congenital vessel connections and assigns venous structures
according to their cardiac connections.

## ImageALCAPA

ImageALCAPA contains 30 preoperative 3D CT angiography volumes from patients
with anomalous left coronary artery arising from the pulmonary artery. The
seven annotated structures are myocardium, left ventricle, right ventricle,
ascending aorta, pulmonary artery, left coronary artery, and right coronary
artery. This source uses coronary targets instead of the two atrial targets in
CHD68 and MM-WHS.

## MM-WHS

MM-WHS is a public whole-heart segmentation benchmark with 60 CT volumes,
including 20 challenge training cases and 40 challenge test cases. Its seven
structures are myocardium, left atrium, left ventricle, right atrium, right
ventricle, ascending aorta, and pulmonary artery. The supplied local
evaluation collection contains masks for all 60 cases; users should verify
the provenance and license of any local mask collection.

## Label map

The released model uses the unified label space:

    0 background
    1 LV
    2 RV
    3 LA
    4 RA
    5 Myo
    6 Ao
    7 PA
    8 LCA
    9 RCA

Losses are masked by source protocol during training. At inference, the source
is known from the case prefix and selects the corresponding source-conditioned
prediction path.
