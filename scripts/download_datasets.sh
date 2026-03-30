#!/bin/bash
# =============================================================================
# MedZFS Dataset Download Script
# =============================================================================
# Downloads and organizes the three benchmark datasets:
#   1. Abdomen MRI (CHAOS Challenge)
#   2. Abdomen CT (Synapse Multi-Organ)
#   3. Cardiac MRI (MS-CMRSeg)
#
# Usage:
#   bash scripts/download_datasets.sh --dataset all --output_dir ./data/raw/
# =============================================================================

set -e

OUTPUT_DIR="./data/raw"
DATASET="all"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "  MedZFS Dataset Download"
echo "=============================================="

download_abd_mri() {
    echo ""
    echo "--- Abdomen MRI (CHAOS Challenge) ---"
    echo "The CHAOS dataset requires registration at:"
    echo "  https://chaos.grand-challenge.org/"
    echo ""
    echo "After downloading, place the data in:"
    echo "  ${OUTPUT_DIR}/abd_mri/"
    echo "    images/  -> NIfTI volumes"
    echo "    masks/   -> NIfTI segmentation masks"
    echo ""
    mkdir -p "${OUTPUT_DIR}/abd_mri/images"
    mkdir -p "${OUTPUT_DIR}/abd_mri/masks"
}

download_abd_ct() {
    echo ""
    echo "--- Abdomen CT (Synapse Multi-Organ) ---"
    echo "The Synapse dataset requires registration at:"
    echo "  https://www.synapse.org/#!Synapse:syn3193805"
    echo ""
    echo "After downloading, place the data in:"
    echo "  ${OUTPUT_DIR}/abd_ct/"
    echo "    images/  -> NIfTI volumes"
    echo "    masks/   -> NIfTI segmentation masks"
    echo ""
    mkdir -p "${OUTPUT_DIR}/abd_ct/images"
    mkdir -p "${OUTPUT_DIR}/abd_ct/masks"
}

download_cmr() {
    echo ""
    echo "--- Cardiac MRI (MS-CMRSeg) ---"
    echo "The MS-CMRSeg dataset is available at:"
    echo "  http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mscmrseg/"
    echo ""
    echo "After downloading, place the data in:"
    echo "  ${OUTPUT_DIR}/cmr/"
    echo "    images/  -> NIfTI volumes"
    echo "    masks/   -> NIfTI segmentation masks"
    echo ""
    mkdir -p "${OUTPUT_DIR}/cmr/images"
    mkdir -p "${OUTPUT_DIR}/cmr/masks"
}

case $DATASET in
    abd_mri) download_abd_mri ;;
    abd_ct)  download_abd_ct ;;
    cmr)     download_cmr ;;
    all)
        download_abd_mri
        download_abd_ct
        download_cmr
        ;;
    *) echo "Unknown dataset: $DATASET. Options: abd_mri, abd_ct, cmr, all"; exit 1 ;;
esac

echo ""
echo "=============================================="
echo "  Directory structure created at: $OUTPUT_DIR"
echo "  Please download datasets manually and place"
echo "  NIfTI files in the appropriate directories."
echo "=============================================="
