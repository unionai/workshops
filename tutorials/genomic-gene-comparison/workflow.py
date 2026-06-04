"""
Gene Comparison Across Species - Compare homologous genes with Carbon + ESMFold.

Pipeline: load homologous gene sequences from different species, score with
Carbon-3B to measure sequence likelihood, compute cross-species similarity,
translate to protein, fold with ESMFold for 3D structure comparison, and
build a phylogenetic tree from sequence divergence.

This tutorial shows the full "DNA to structure" journey: how the same gene
evolves across species, and whether sequence changes translate to structural
differences.

Usage:
    # Default (Insulin gene across 6 species)
    flyte run --local --tui workflow.py pipeline

    # Different gene set
    flyte run --local --tui workflow.py pipeline --gene_set "hemoglobin"

    # Remote (GPU for Carbon + ESMFold)
    flyte run workflow.py pipeline
"""

import json
import logging
import math
import os
import tempfile

import flyte
import flyte.io
import flyte.report
from config import cpu_env, gpu_env

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ------------------------------------------------------------------
# Homologous gene sets - same gene across species
# ------------------------------------------------------------------
# Full-length coding sequences from NCBI RefSeq (stop codon excluded).

GENE_SETS = {
    "insulin": {
        "gene_name": "Insulin",
        "description": "Insulin regulates blood sugar in all vertebrates. Highly conserved across 500M+ years of evolution - even fish insulin can lower blood sugar in humans. Comparing across species reveals which regions are functionally essential (conserved) vs free to drift.",
        "sequences": {
            "Human": {
                "dna": "ATGGCCCTGTGGATGCGCCTCCTGCCCCTGCTGGCGCTGCTGGCCCTCTGGGGACCTGACCCAGCCGCAGCCTTTGTGAACCAACACCTGTGCGGCTCACACCTGGTGGAAGCTCTCTACCTAGTGTGCGGGGAACGAGGCTTCTTCTACACACCCAAGACCCGCCGGGAGGCAGAGGACCTGCAGGTGGGGCAGGTGGAGCTGGGCGGGGGCCCTGGTGCAGGCAGCCTGCAGCCCTTGGCCCTGGAGGGGTCCCTGCAGAAGCGTGGCATTGTGGAACAATGCTGTACCAGCATCTGCTCCCTCTACCAGCTGGAGAACTACTGCAAC",
                "common_name": "Homo sapiens",
            },
            "Mouse": {
                "dna": "ATGGCCCTGTGGATGCGCTTCCTGCCCCTGCTGGCCCTGCTCTTCCTCTGGGAGTCCCACCCCACCCAGGCTTTTGTCAAGCAGCACCTTTGTGGTTCCCACCTGGTGGAGGCTCTCTACCTGGTGTGTGGGGAGCGTGGCTTCTTCTACACACCCATGTCCCGCCGTGAAGTGGAGGACCCACAAGTGGCACAACTGGAGCTGGGTGGAGGCCCGGGAGCAGGTGACCTTCAGACCTTGGCACTGGAGGTGGCCCAGCAGAAGCGTGGCATTGTAGATCAGTGCTGCACCAGCATCTGCTCCCTCTACCAGCTGGAGAACTACTGCAAC",
                "common_name": "Mus musculus",
            },
            "Chicken": {
                "dna": "ATGGCTCTCTGGATCCGATCACTGCCTCTTCTGGCTCTCCTTGTCTTTTCTGGCCCTGGAACCAGCTATGCAGCTGCCAACCAGCACCTCTGTGGCTCCCACTTGGTGGAGGCTCTCTACCTGGTGTGTGGAGAGCGTGGCTTCTTCTACTCCCCCAAAGCCCGACGGGATGTCGAGCAGCCCCTAGTGAGCAGTCCCTTGCGTGGCGAGGCAGGAGTGCTGCCTTTCCAGCAGGAGGAATACGAGAAAGTCAAGCGAGGGATTGTTGAGCAATGCTGCCATAACACGTGTTCCCTCTACCAACTGGAGAACTACTGCAAC",
                "common_name": "Gallus gallus",
            },
            "Zebrafish": {
                "dna": "ATGGCAGTGTGGCTTCAGGCTGGTGCTCTGTTGGTCCTGTTGGTCGTGTCCAGTGTAAGCACTAACCCAGGCACACCGCAGCACCTGTGTGGATCTCATCTGGTCGATGCCCTTTATCTGGTCTGTGGCCCAACAGGCTTCTTCTACAACCCCAAGAGAGACGTTGAGCCCCTTCTGGGTTTCCTTCCTCCTAAATCTGCCCAGGAAACTGAGGTGGCTGACTTTGCATTTAAAGATCATGCCGAGCTGATAAGGAAGAGAGGCATTGTAGAGCAGTGCTGCCACAAACCCTGCAGCATCTTTGAGCTGCAGAACTACTGTAAC",
                "common_name": "Danio rerio",
            },
            "Frog": {
                "dna": "ATGGCTCTATGGATGCAGTGTCTGCCCCTGGTTCTTGTCCTCTTTTTCTCTACACCCAACACCGAAGCTCTAGTTAACCAGCACTTGTGTGGGTCTCACCTGGTAGAAGCCCTGTACTTAGTATGTGGGGATCGAGGCTTCTTCTACTACCCTAAGGTCAAACGGGACATGGAACAAGCACTTGTCAGTGGACCCCAGGATAATGAGTTGGATGGAATGCAGCTCCAGCCTCAGGAGTATCAGAAAATGAAGAGGGGGATTGTGGAGCAATGTTGCCACAGCACATGTTCTCTCTTCCAGCTGGAGAGTTACTGCAAC",
                "common_name": "Xenopus laevis",
            },
            "Cow": {
                "dna": "ATGGCCCTGTGGACACGCCTGGCGCCCCTGCTGGCCCTGCTGGCGCTCTGGGCCCCCGCCCCGGCCCGCGCCTTCGTCAACCAGCATCTGTGTGGCTCCCACCTGGTGGAGGCGCTGTACCTGGTGTGCGGAGAGCGCGGCTTCTTCTACACGCCCAAGGCCCGCCGGGAGGTGGAGGGCCCCCAGGTGGGGGCGCTGGAGCTGGCCGGAGGCCCGGGCGCGGGCGGCCTGGAGGGGCCCCCGCAGAAGCGTGGCATCGTGGAGCAGTGCTGTGCCAGCGTCTGCTCGCTCTACCAGCTGGAGAACTACTGTAAC",
                "common_name": "Bos taurus",
            },
        },
    },
    "hemoglobin": {
        "gene_name": "Hemoglobin Beta",
        "description": "Beta-globin carries oxygen from lungs to tissues. The most studied gene in molecular evolution - sequence differences power the 'molecular clock' hypothesis. Sickle cell mutation (E6V) in humans shows how a single base change creates devastating disease.",
        "sequences": {
            "Human": {
                "dna": "ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCCTGAGAACTTCAGGCTCCTGGGCAACGTGCTGGTCTGTGTGCTGGCCCATCACTTTGGCAAAGAATTCACCCCACCAGTGCAGGCTGCCTATCAGAAAGTGGTGGCTGGTGTGGCTAATGCCCTGGCCCACAAGTATCAC",
                "common_name": "Homo sapiens",
            },
            "Mouse": {
                "dna": "ATGGTGCACCTGACTGATGCTGAGAAGGCTGCTGTCTCTGGCCTGTGGGGAAAGGTGAACGCCGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTTGTCTACCCTTGGACCCAGCGGTACTTTGATAGCTTTGGAGACCTATCCTCTGCCTCTGCTATCATGGGTAATGCCAAAGTGAAGGCCCATGGCAAGAAAGTGATAACTGCCTTTAACGATGGCCTGAATCACTTGGACAGCCTCAAGGGCACCTTTGCCAGCCTCAGTGAGCTCCACTGTGACAAGCTGCATGTGGATCCTGAGAACTTCAGGCTCCTGGGCAATATGATCGTGATTGTGCTGGGCCACCACCTGGGCAAGGATTTCACCCCCGCTGCACAGGCTGCCTTCCAGAAGGTGGTGGCTGGAGTGGCTGCTGCCCTGGCTCACAAGTACCAC",
                "common_name": "Mus musculus",
            },
            "Chicken": {
                "dna": "ATGGTGCACTGGACTGCTGAGGAGAAGCAGCTCATCACCGGCCTCTGGGGCAAGGTCAATGTGGCCGAATGTGGGGCTGAAGCCCTGGCCAGGCTGCTGATCGTCTACCCCTGGACCCAGAGGTTCTTTGCGTCCTTTGGGAACCTCTCCAGCCCCACTGCCATCCTTGGCAACCCCATGGTCCGCGCCCATGGCAAGAAAGTGCTCACCTCCTTTGGGGATGCTGTGAAGAACCTGGACAACATCAAGAACACCTTCTCCCAACTGTCCGAACTGCATTGTGACAAGCTGCATGTGGACCCCGAGAACTTCAGGCTCCTGGGTGACATCCTCATCATTGTCCTGGCCGCCCACTTCAGCAAGGACTTCACTCCTGAATGCCAGGCTGCCTGGCAGAAGCTGGTCCGCGTGGTGGCCCATGCCCTGGCTCGCAAGTACCAC",
                "common_name": "Gallus gallus",
            },
            "Zebrafish": {
                "dna": "ATGGTTGAGTGGACAGATGCCGAGCGCACAGCCATCCTTGGCCTGTGGGGAAAGCTCAATATCGATGAAATCGGACCTCAGGCCCTATCCAGATGTCTGATCGTGTATCCCTGGACTCAGAGATATTTCGCCACATTCGGCAACCTGTCAAGCCCCGCTGCGATCATGGGTAACCCCAAAGTGGCAGCTCATGGGAGGACTGTGATGGGAGGTCTTGAGAGAGCCATCAAGAACATGGACAACGTCAAGAACACCTATGCCGCCCTCAGTGTGATGCACTCTGAGAAACTGCATGTGGATCCCGACAACTTCAGGCTTCTCGCTGATTGCATCACCGTTTGCGCTGCCATGAAGTTCGGCCAAGCTGGTTTCAATGCTGATGTCCAGGAGGCCTGGCAGAAGTTTCTGGCTGTGGTCGTTTCTGCTCTGTGCAGACAGTACCAC",
                "common_name": "Danio rerio",
            },
            "Frog": {
                "dna": "ATGGTTCATTGGACAGCTGAAGAGAAGGCCGCCATCACCTCTGTGTGGCAGGAGGTCAACCAGGAGCAAGATGGCCATGATGCACTCACAAGGCTGCTGGTTGTGTACCCCTGGACCCAGAGATACTTCAGCAGTTTTGGAAATCTCGGTAATGCCACAGCTATTGCTGGAAATGTCAAGGTGCGTGCCCATGGCAAGAAGGTTCTTTCAGCTGTTGGTGATGCCATCGCCCATCTTGACAACGTGAAGGGAACTCTCCATGACCTCAGTGTGGTCCACGCCTTCAAGCTCTATGTGGATCCTGAGAACTTCAAGCGTCTTGGTGAAGTTCTGGTCATTGTCTTGGCTTCCAAACTGGGATCAGCCTTTACTCCTCAAGTCCAGGGAGCCTGGGAGAAATTTGTTGCTGTTCTGGTTGATGCCCTCAGCCAAGGATACAAC",
                "common_name": "Xenopus laevis",
            },
            "Cow": {
                "dna": "ATGCTGACTGCTGAGGAGAAGGCTGCCGTCACCGCCTTTTGGGGCAAGGTGAAAGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTTGTCTACCCCTGGACTCAGAGGTTCTTTGAGTCCTTTGGGGACTTGTCCACTGCTGATGCTGTTATGAACAACCCTAAGGTGAAGGCCCATGGCAAGAAGGTGCTAGATTCCTTTAGTAATGGCATGAAGCATCTCGATGACCTCAAGGGCACCTTTGCTGCGCTGAGTGAGCTGCACTGTGATAAGCTGCATGTGGATCCTGAGAACTTCAAGCTCCTGGGCAACGTGCTAGTGGTTGTGCTGGCTCGCAATTTTGGCAAGGAATTCACCCCGGTGCTGCAGGCTGACTTTCAGAAGGTGGTGGCTGGTGTGGCCAATGCCCTGGCCCACAGATATCAT",
                "common_name": "Bos taurus",
            },
        },
    },
    "p53": {
        "gene_name": "p53 (TP53)",
        "description": "The 'guardian of the genome' - p53 detects DNA damage and triggers repair or cell death. Mutated in >50% of human cancers. Elephants have 20 copies of p53 (humans have 1), which may explain their extremely low cancer rates despite their size (Peto's paradox).",
        "sequences": {
            "Human": {
                "dna": "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAGACCTATGGAAACTACTTCCTGAAAACAACGTTCTGTCCCCCTTGCCGTCCCAAGCAATGGATGATTTGATGCTGTCCCCGGACGATATTGAACAATGGTTCACTGAAGACCCAGGTCCAGATGAAGCTCCCAGAATGCCAGAGGCTGCTCCCCCCGTGGCCCCTGCACCAGCAGCTCCTACACCGGCGGCCCCTGCACCAGCCCCCTCCTGGCCCCTGTCATCTTCTGTCCCTTCCCAGAAAACCTACCAGGGCAGCTACGGTTTCCGTCTGGGCTTCTTGCATTCTGGGACAGCCAAGTCTGTGACTTGCACGTACTCCCCTGCCCTCAACAAGATGTTTTGCCAACTGGCCAAGACCTGCCCTGTGCAGCTGTGGGTTGATTCCACACCCCCGCCCGGCACCCGCGTCCGCGCCATGGCCATCTACAAGCAGTCACAGCACATGACGGAGGTTGTGAGGCGCTGCCCCCACCATGAGCGCTGCTCAGATAGCGATGGTCTGGCCCCTCCTCAGCATCTTATCCGAGTGGAAGGAAATTTGCGTGTGGAGTATTTGGATGACAGAAACACTTTTCGACATAGTGTGGTGGTGCCCTATGAGCCGCCTGAGGTTGGCTCTGACTGTACCACCATCCACTACAACTACATGTGTAACAGTTCCTGCATGGGCGGCATGAACCGGAGGCCCATCCTCACCATCATCACACTGGAAGACTCCAGTGGTAATCTACTGGGACGGAACAGCTTTGAGGTGCGTGTTTGTGCCTGTCCTGGGAGAGACCGGCGCACAGAGGAAGAGAATCTCCGCAAGAAAGGGGAGCCTCACCACGAGCTGCCCCCAGGGAGCACTAAGCGAGCACTGCCCAACAACACCAGCTCCTCTCCCCAGCCAAAGAAGAAACCACTGGATGGAGAATATTTCACCCTTCAGATCCGTGGGCGTGAGCGCTTCGAGATGTTCCGAGAGCTGAATGAGGCCTTGGAACTCAAGGATGCCCAGGCTGGGAAGGAGCCAGGGGGGAGCAGGGCTCACTCCAGCCACCTGAAGTCCAAAAAGGGTCAGTCTACCTCCCGCCATAAAAAACTCATGTTCAAGACAGAAGGGCCTGACTCAGAC",
                "common_name": "Homo sapiens",
            },
            "Mouse": {
                "dna": "ATGACTGCCATGGAGGAGTCACAGTCGGATATCAGCCTCGAGCTCCCTCTGAGCCAGGAGACATTTTCAGGCTTATGGAAACTACTTCCTCCAGAAGATATCCTGCCATCACCTCACTGCATGGACGATCTGTTGCTGCCCCAGGATGTTGAGGAGTTTTTTGAAGGCCCAAGTGAAGCCCTCCGAGTGTCAGGAGCTCCTGCAGCACAGGACCCTGTCACCGAGACCCCTGGGCCAGTGGCCCCTGCCCCAGCCACTCCATGGCCCCTGTCATCTTTTGTCCCTTCTCAAAAAACTTACCAGGGCAACTATGGCTTCCACCTGGGCTTCCTGCAGTCTGGGACAGCCAAGTCTGTTATGTGCACGTACTCTCCTCCCCTCAATAAGCTATTCTGCCAGCTGGCGAAGACGTGCCCTGTGCAGTTGTGGGTCAGCGCCACACCTCCAGCTGGGAGCCGTGTCCGCGCCATGGCCATCTACAAGAAGTCACAGCACATGACGGAGGTCGTGAGACGCTGCCCCCACCATGAGCGCTGCTCCGATGGTGATGGCCTGGCTCCTCCCCAGCATCTTATCCGGGTGGAAGGAAATTTGTATCCCGAGTATCTGGAAGACAGGCAGACTTTTCGCCACAGCGTGGTGGTACCTTATGAGCCACCCGAGGCCGGCTCTGAGTATACCACCATCCACTACAAGTACATGTGTAATAGCTCCTGCATGGGGGGCATGAACCGCCGACCTATCCTTACCATCATCACACTGGAAGACTCCAGTGGGAACCTTCTGGGACGGGACAGCTTTGAGGTTCGTGTTTGTGCCTGCCCTGGGAGAGACCGCCGTACAGAAGAAGAAAATTTCCGCAAAAAGGAAGTCCTTTGCCCTGAACTGCCCCCAGGGAGCGCAAAGAGAGCGCTGCCCACCTGCACAAGCGCCTCTCCCCCGCAAAAGAAAAAACCACTTGATGGAGAGTATTTCACCCTCAAGATCCGCGGGCGTAAACGCTTCGAGATGTTCCGGGAGCTGAATGAGGCCTTAGAGTTAAAGGATGCCCATGCTACAGAGGAGTCTGGAGACAGCAGGGCTCACTCCAGCTACCTGAAGACCAAGAAGGGCCAGTCTACTTCCCGCCATAAAAAAACAATGGTCAAGAAAGTGGGGCCTGACTCAGAC",
                "common_name": "Mus musculus",
            },
            "Chicken": {
                "dna": "ATGGCGGAGGAGATGGAACCATTGCTGGAACCCACTGAGGTCTTCATGGACCTCTGGAGCATGCTCCCCTATAGCATGCAACAGCTGCCCCTCCCTGAGGATCACAGCAACTGGCAGGAGCTGAGCCCCCTGGAACCCAGCGACCCCCCCCCACCACCGCCACCACCACCTCTGCCATTGGCCGCCGCCGCCCCCCCCCCATTAAACCCCCCCACCCCCCCCCGCGCTGCCCCCTCCCCGGTGGTCCCATCCACGGAGGATTATGGGGGGGACTTCGACTTCCGGGTGGGGTTCGTGGAGGCGGGCACAGCCAAATCGGTCACCTGCACTTACTCCCCGGTGCTGAATAAGGTCTATTGCCGCCTGGCCAAGCCGTGCCCGGTGCAGGTGAGGGTGGGGGTGGCGCCCCCCCCCGGTTCCTCCCTCCGCGCCGTGGCCGTCTATAAGAAATCAGAGCACGTGGCCGAAGTGGTGCGGCGCTGCCCCCACCACGAGCGCTGCGGGGGGGGCACCGACGGCCTGGCCCCCGCACAGCACCTCATCCGGGTGGAGGGGAACCCCCAGGCGCGTTACCACGACGACGAGACCACCAAACGGCACAGCGTCGTCGTCCCCTATGAGCCCCCCGAGGTGGGCTCTGACTGTACCACGGTGCTGTACAACTTCATGTGCAACAGTTCCTGCATGGGGGGGATGAACCGCCGCCCCATCCTCACCATCCTTACACTGGAGGGGCCGGGGGGGCAGCTGTTGGGGCGGCGCTGCTTCGAGGTGCGCGTGTGCGCATGTCCGGGGAGGGACCGCAAGATCGAGGAGGAGAACTTCCGCAAGAGGGGCGGGGCCGGGGGCGTGGCTAAGCGAGCCATGTCGCCCCCAACCGAAGCCCCCGAGCCCCCCAAGAAGCGCGTGCTGAACCCCGACAATGAGATATTCTACCTGCAGGTGCGCGGGCGCCGCCGCTATGAGATGCTGAAGGAGATCAATGAGGCGCTGCAGCTCGCCGAGGGGGGGTCCGCACCGCGGCCTTCCAAAGGCCGCCGTGTGAAGGTGGAGGGACCCCAACCCAGCTGCGGGAAGAAACTGCTGCAAAAAGGCTCGGAC",
                "common_name": "Gallus gallus",
            },
            "Zebrafish": {
                "dna": "ATGGCGCAAAACGACAGCCAAGAGTTCGCGGAGCTCTGGGAGAAGAATTTGATTATTCAGCCCCCAGGTGGTGGCTCTTGCTGGGACATCATTAATGATGAGGAGTACTTGCCGGGATCGTTTGACCCCAATTTTTTTGAAAATGTGCTTGAAGAACAGCCTCAGCCATCCACTCTCCCACCAACATCCACTGTTCCGGAGACAAGCGACTATCCCGGCGATCATGGATTTAGGCTCAGGTTCCCGCAGTCTGGCACAGCAAAATCTGTAACTTGCACTTATTCACCGGACCTGAATAAACTCTTCTGTCAGCTGGCAAAAACTTGCCCCGTTCAAATGGTGGTGGACGTTGCCCCTCCACAGGGCTCCGTGGTTCGAGCCACTGCCATCTATAAGAAGTCCGAGCATGTGGCTGAAGTGGTCCGCAGATGCCCCCATCATGAGCGAACCCCGGATGGAGATAACTTGGCGCCTGCTGGTCATTTGATAAGAGTGGAGGGCAATCAGCGAGCAAATTACAGGGAAGATAACATCACTTTAAGGCATAGTGTTTTTGTCCCATATGAAGCACCACAGCTTGGTGCTGAATGGACAACTGTGCTACTAAACTACATGTGCAATAGCAGCTGCATGGGGGGGATGAACCGCAGGCCCATCCTCACAATCATCACTCTGGAGACTCAGGAAGGTCAGTTGCTGGGCCGGAGGTCTTTTGAGGTGCGTGTGTGTGCATGTCCAGGCAGAGACAGGAAAACTGAGGAGAGCAACTTCAAGAAAGACCAAGAGACCAAAACCATGGCCAAAACCACCACTGGGACCAAACGTAGTTTGGTGAAAGAATCTTCTTCAGCTACATTACGACCTGAGGGGAGCAAAAAGGCCAAGGGCTCCAGCAGCGATGAGGAGATCTTTACCCTGCAGGTGAGGGGCAGGGAGCGTTATGAAATTTTAAAGAAATTGAACGACAGTCTGGAGTTAAGTGATGTGGTGCCTGCCTCAGATGCTGAAAAGTATCGTCAGAAATTCATGACAAAAAACAAAAAAGAGAATCGTGAATCATCTGAGCCCAAACAGGGAAAGAAGCTGATGGTGAAGGACGAAGGAAGAAGCGACTCTGAT",
                "common_name": "Danio rerio",
            },
            "Elephant": {
                "dna": "ATGGAGGAGCCCCAGTCAGATCTCAGCACCGAGCTCCCTCTGAGTCAAGAGACGTTTTCATACTTATGGGAACTCCTTCCTGAGAATCCGGTTCTGTCCCCCACACTACCCCCGGCAGTGGAGGTCATGGACGATCTGCTACTCTCAGAAGACACTGCAAACTGGCTAGAAAGCCAAGTTGAGGCTCAGGGAATGTCCACAACCCCTGCACCAGCCACCCCTACACCGGTGGCCCCCGCACCAGCCACCTCCTGGACCCTGTCATCTTCCGTCCCTTCCCAAAAGACCTACCCTGGCACCTATGGTTTCCGTCTGGGCTTCCTACATTCTGGGACAGCCAAGTCCGTCACCTGCACGTACTCCCCTGACCTTAACAAGCTGTTTTGCCAGCTGGCAAAAACCTGCCCAGTGCAGCTGTGGGTCGCCTCACCACCCCCGCCCGGCACCCGTGTTCGCACCATGGCCATCTACAAGAAGTCAGAGCATATGACGGAGGTCGTCAAGCGCTGCCCCCACCATGAGCGCTGCTCTGACTCTAGCGATGGCCTGGCCCCTCCTCAGCACCTCATCCGGGTGGAAGGAAACCTGCGTGCTGAGTATCTGGAGGACAGCATCACTCTCCGACACAGTGTGGTGGTGCCCTACGAGCCGCCCGAGGTTGGGTCTGACTGTACCACCATCCACTTCAACTTCATGTGTAACAGCTCCTGCATGGGGGGCATGAACCGGCGGCCCATCCTCACCATCATCACACTGGAAGACTCCAGTGGTAATCTGCTGGGACGTAACAGCTTTGAGGTGCGCATTTGTGCCTGTCCTGGAAGAGACAGACGTACAGAAGAAGAAAATTTCCACAAGAAGGGAGAGCCTTGCCCAGAGCCGCCACCCCCTGGGAGGAGCACTAAGCGAGCACTGCCCACCAACACCAGCTCCTCTACCCAGCCAAAGAAGAAGCCACTGGATGAAGAATATTTCACCCTTCAGATCCGTGGGCGTGAACGCTTCAAGATGTTCCTAGAGCTAAATGAGGCCTTGGAGCTGAAGGATGCCCAGGCTGGGAAGGAGCCAGAGGGGAGCCGGGCTCACTCCAGCCCTTCGAAGTCTAAGAAGGGACAGTCTACCTCCCGCCATAAAAAACCAATGTTCAAGAGAGAGGGACCTGACTCAGAC",
                "common_name": "Loxodonta africana",
            },
            "Dog": {
                "dna": "ATGGAGGAGTCGCAGTCAGAGCTCAATATCGACCCCCCTCTGAGCCAGGAGACATTTTCAGAATTGTGGAACCTGCTTCCTGAAAACAATGTTCTGTCTTCGGAGCTGTGCCCAGCAGTGGATGAGCTGCTGCTCCCAGAGAGCGTCGTGAACTGGCTAGACGAAGACTCAGATGATGCTCCCAGGATGCCAGCCACTTCTGCCCCCACAGCCCCTGGACCGGCCCCCTCGTGGCCCCTATCATCCTCTGTCCCTTCCCCGAAGACCTACCCTGGCACCTATGGGTTCCGTTTGGGGTTCCTGCATTCCGGGACAGCCAAGTCTGTTACTTGGACGTACTCCCCTCTCCTCAACAAGTTGTTTTGCCAGCTGGCGAAGACCTGCCCCGTGCAGCTGTGGGTCAGCTCCCCACCCCCACCCAATACCTGCGTCCGCGCTATGGCCATCTATAAGAAGTCGGAGTTCGTGACCGAGGTTGTGCGGCGCTGCCCCCACCATGAACGCTGCTCTGACAGTAGTGACGGTCTTGCCCCTCCTCAGCATCTCATCCGAGTGGAAGGAAATTTGCGGGCCAAGTACCTGGACGACAGAAACACTTTTCGACACAGTGTGGTGGTGCCTTATGAGCCACCCGAGGTTGGCTCTGACTATACCACCATCCACTACAACTACATGTGTAACAGTTCCTGCATGGGAGGCATGAACCGGCGGCCCATCCTCACTATCATCACCCTGGAAGACTCCAGTGGAAACGTGCTGGGACGCAACAGCTTTGAGGTACGCGTTTGTGCCTGTCCCGGGAGAGACCGCCGGACTGAGGAGGAGAATTTCCACAAGAAGGGGGAGCCTTGTCCTGAGCCACCCCCCGGGAGTACCAAGCGAGCACTGCCTCCCAGCACCAGCTCCTCTCCCCCGCAAAAGAAGAAGCCACTAGATGGAGAATATTTCACCCTTCAGATCCGTGGGCGTGAACGCTATGAGATGTTCAGGAATCTGAATGAAGCCTTGGAGCTGAAGGATGCCCAGAGTGGAAAGGAGCCAGGGGGAAGCAGGGCTCACTCCAGCCACCTGAAGGCAAAGAAGGGGCAATCTACCTCTCGCCATAAAAAACTGATGTTCAAGAGAGAAGGGCTTGACTCAGAC",
                "common_name": "Canis lupus familiaris",
            },
        },
    },
}

# Standard genetic code
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L",
    "CTA": "L", "CTG": "L", "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "TCT": "S", "TCC": "S",
    "TCA": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A",
    "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W", "CGT": "R", "CGC": "R",
    "CGA": "R", "CGG": "R", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

BASE_COLORS = {"A": "#2ecc71", "T": "#e74c3c", "G": "#f39c12", "C": "#3498db"}


def _translate(dna: str) -> str:
    """Translate DNA to protein in reading frame 0."""
    dna = dna.upper()
    protein = []
    for i in range(0, len(dna) - 2, 3):
        codon = dna[i:i + 3]
        aa = CODON_TABLE.get(codon, "X")
        if aa == "*":
            break
        protein.append(aa)
    return "".join(protein)


def _gc_content(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(1 for b in seq.upper() if b in "GC") / len(seq)


def _sequence_identity(seq1: str, seq2: str, match: int = 2, mismatch: int = -1, gap: int = -2) -> float:
    """Percent identity via Needleman-Wunsch global alignment."""
    if not seq1 or not seq2:
        return 0.0
    n, m = len(seq1), len(seq2)

    # Build score matrix
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if seq1[i - 1] == seq2[j - 1] else mismatch
            dp[i][j] = max(dp[i - 1][j - 1] + s, dp[i - 1][j] + gap, dp[i][j - 1] + gap)

    # Traceback to count matches and alignment length
    i, j = n, m
    matches = 0
    aligned = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            s = match if seq1[i - 1] == seq2[j - 1] else mismatch
            if dp[i][j] == dp[i - 1][j - 1] + s:
                if seq1[i - 1] == seq2[j - 1]:
                    matches += 1
                aligned += 1
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + gap:
            aligned += 1
            i -= 1
        else:
            aligned += 1
            j -= 1

    return matches / aligned if aligned else 0.0


# ------------------------------------------------------------------
# Report styling
# ------------------------------------------------------------------

REPORT_CSS = """
<style>
  .report { font-family: system-ui, -apple-system, sans-serif; max-width: 960px; margin: 0 auto; color: #1a1a2e; }
  .report h2 { color: #1e3a5f; border-bottom: 2px solid #2563eb; padding-bottom: 8px; margin-top: 24px; }
  .report h3 { color: #1e40af; margin-top: 20px; }
  .report .card { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .report .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 12px 0; }
  .report .stat { background: #fff; border: 1px solid #dbeafe; border-radius: 6px; padding: 12px; text-align: center; }
  .report .stat .value { font-size: 1.5em; font-weight: 700; color: #1e3a5f; }
  .report .stat .label { font-size: 0.85em; color: #6c757d; margin-top: 4px; }
  .report table { border-collapse: collapse; width: 100%; margin: 12px 0; }
  .report th { background: #1e3a5f; color: #fff; padding: 10px 14px; text-align: left; font-weight: 600; }
  .report td { padding: 8px 14px; border-bottom: 1px solid #dbeafe; }
  .report tr:nth-child(even) { background: #eff6ff; }
  .report .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }
  .report .badge-success { background: #d1fae5; color: #065f46; }
  .report .badge-warning { background: #fef3c7; color: #92400e; }
  .report .badge-danger { background: #fee2e2; color: #991b1b; }
  .report .badge-info { background: #dbeafe; color: #1e40af; }
  .report .chart-container { background: #fff; border: 1px solid #dbeafe; border-radius: 8px; padding: 16px; margin: 16px 0; }
  .report .note { background: #eff6ff; border-left: 4px solid #2563eb; padding: 10px 14px; border-radius: 4px; margin: 12px 0; font-size: 0.9em; }
  .report .structure-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; margin: 12px 0; }
</style>
"""


def _wrap_report(html: str) -> str:
    return f'{REPORT_CSS}<div class="report">{html}</div>'


# ------------------------------------------------------------------
# SVG chart helpers
# ------------------------------------------------------------------

def _make_heatmap(
    matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    title: str = "",
    width: int = 600,
    height: int = 500,
    value_format: str = ".1f",
    color_scale: str = "blue",
) -> str:
    """Generate an SVG heatmap."""
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if matrix else 0
    if not n_rows or not n_cols:
        return ""

    show_values = n_rows <= 10 and n_cols <= 10
    flat = [v for row in matrix for v in row]
    v_min = min(flat)
    v_max = max(flat)
    v_range = v_max - v_min or 1

    if color_scale == "blue":
        def get_color(v):
            t = (v - v_min) / v_range
            r = int(255 - t * (255 - 30))
            g = int(255 - t * (255 - 58))
            b = int(255 - t * (255 - 95))
            return f"rgb({r},{g},{b})"
    else:  # green
        def get_color(v):
            t = (v - v_min) / v_range
            r = int(255 - t * (255 - 6))
            g = int(255 - t * (255 - 95))
            b = int(255 - t * (255 - 70))
            return f"rgb({r},{g},{b})"

    ml = max(80, max(len(l) for l in row_labels) * 7 + 10) if row_labels else 80
    mr = 20
    mt = 80
    mb = 20
    cw = width - ml - mr
    ch = height - mt - mb
    cell_w = cw / n_cols
    cell_h = ch / n_rows

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    if title:
        svg.append(f'<text x="{width / 2}" y="22" text-anchor="middle" font-size="14" font-weight="600" fill="#1a1a2e">{title}</text>')

    for j, label in enumerate(col_labels):
        cx = ml + j * cell_w + cell_w / 2
        svg.append(f'<text x="{cx:.1f}" y="{mt - 8}" text-anchor="start" font-size="10" fill="#374151" transform="rotate(-45, {cx:.1f}, {mt - 8})">{label}</text>')

    for i, row_label in enumerate(row_labels):
        ry = mt + i * cell_h + cell_h / 2
        svg.append(f'<text x="{ml - 8}" y="{ry + 4:.1f}" text-anchor="end" font-size="10" fill="#374151">{row_label}</text>')
        for j in range(n_cols):
            val = matrix[i][j]
            color = get_color(val)
            cx = ml + j * cell_w
            cy = mt + i * cell_h
            svg.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{color}" stroke="#fff" stroke-width="1"/>')
            if show_values:
                t = (val - v_min) / v_range
                text_color = "#fff" if t > 0.55 else "#1a1a2e"
                fs = min(10, int(cell_w / 4), int(cell_h / 2.5))
                fs = max(7, fs)
                svg.append(f'<text x="{cx + cell_w / 2:.1f}" y="{cy + cell_h / 2 + 3:.1f}" text-anchor="middle" font-size="{fs}" fill="{text_color}">{val:{value_format}}</text>')

    svg.append("</svg>")
    return "\n".join(svg)


def _make_dendrogram(
    names: list[str],
    matrix: list[list[float]],
    title: str = "",
    width: int = 700,
    height: int = 350,
    color: str = "#2563eb",
) -> str:
    """Generate an SVG dendrogram from a similarity matrix using UPGMA."""
    n = len(names)
    if n < 2:
        return ""

    dist = [[1.0 - matrix[i][j] for j in range(n)] for i in range(n)]

    clusters = [{"members": [i], "height": 0.0, "left": None, "right": None} for i in range(n)]
    active = list(range(n))

    while len(active) > 1:
        best_d = float("inf")
        bi, bj = 0, 1
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                ci, cj = active[ii], active[jj]
                d = 0
                count = 0
                for mi in clusters[ci]["members"]:
                    for mj in clusters[cj]["members"]:
                        d += dist[mi][mj]
                        count += 1
                avg_d = d / count if count else 0
                if avg_d < best_d:
                    best_d = avg_d
                    bi, bj = ii, jj

        ci, cj = active[bi], active[bj]
        new_cluster = {
            "members": clusters[ci]["members"] + clusters[cj]["members"],
            "height": best_d,
            "left": clusters[ci],
            "right": clusters[cj],
        }
        clusters.append(new_cluster)
        new_idx = len(clusters) - 1
        active.pop(bj)
        active.pop(bi)
        active.append(new_idx)

    root = clusters[active[0]]

    max_label_len = max((len(n) for n in names), default=0)
    ml, mr, mt, mb = max(50, max_label_len * 5 + 10), 30, 40, 80
    cw = width - ml - mr
    ch = height - mt - mb
    max_h = root["height"] or 1

    leaf_positions = {}
    leaf_counter = [0]

    def assign_leaves(node):
        if node["left"] is None and node["right"] is None:
            leaf_positions[node["members"][0]] = leaf_counter[0]
            leaf_counter[0] += 1
        else:
            if node["left"]:
                assign_leaves(node["left"])
            if node["right"]:
                assign_leaves(node["right"])

    assign_leaves(root)
    n_leaves = len(leaf_positions)
    leaf_spacing = cw / max(n_leaves - 1, 1)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    if title:
        svg.append(f'<text x="{width / 2}" y="22" text-anchor="middle" font-size="13" font-weight="600" fill="#1a1a2e">{title}</text>')

    def get_x(node):
        if node["left"] is None and node["right"] is None:
            return ml + leaf_positions[node["members"][0]] * leaf_spacing
        return (get_x(node["left"]) + get_x(node["right"])) / 2

    def get_y(h):
        return mt + ch - (h / max_h) * ch

    def draw_node(node):
        if node["left"] is None and node["right"] is None:
            return
        lx = get_x(node["left"])
        rx = get_x(node["right"])
        ly = get_y(node["left"]["height"])
        ry = get_y(node["right"]["height"])
        my = get_y(node["height"])

        svg.append(f'<line x1="{lx:.1f}" y1="{ly:.1f}" x2="{lx:.1f}" y2="{my:.1f}" stroke="{color}" stroke-width="2"/>')
        svg.append(f'<line x1="{rx:.1f}" y1="{ry:.1f}" x2="{rx:.1f}" y2="{my:.1f}" stroke="{color}" stroke-width="2"/>')
        svg.append(f'<line x1="{lx:.1f}" y1="{my:.1f}" x2="{rx:.1f}" y2="{my:.1f}" stroke="{color}" stroke-width="2"/>')

        if node["left"]:
            draw_node(node["left"])
        if node["right"]:
            draw_node(node["right"])

    draw_node(root)

    for idx, pos in leaf_positions.items():
        x = ml + pos * leaf_spacing
        svg.append(
            f'<text x="{x:.1f}" y="{mt + ch + 14}" text-anchor="start" font-size="10" fill="#374151" '
            f'transform="rotate(40, {x:.1f}, {mt + ch + 14})">{names[idx]}</text>'
        )

    for i in range(5):
        d = max_h * i / 4
        y = get_y(d)
        svg.append(f'<text x="{ml - 4}" y="{y + 3:.1f}" text-anchor="end" font-size="9" fill="#9ca3af">{d:.3f}</text>')

    svg.append("</svg>")
    return "\n".join(svg)


def _make_bar_chart(
    labels: list[str],
    series: dict[str, list[float]],
    title: str = "",
    colors: list[str] | None = None,
    width: int = 700,
    height: int = 300,
    value_format: str = ".1f",
) -> str:
    """Generate an SVG grouped bar chart."""
    if not labels:
        return ""

    default_colors = ["#2563eb", "#059669", "#f59e0b", "#dc2626", "#7c3aed"]
    colors = colors or default_colors

    ml, mr, mt, mb = 60, 20, 40, 80
    cw = width - ml - mr
    ch = height - mt - mb

    all_vals = [v for vals in series.values() for v in vals]
    y_min = min(all_vals) if all_vals else 0
    y_max = max(all_vals) if all_vals else 1
    if y_min >= 0:
        y_min_plot = 0
        y_max_plot = y_max * 1.15 or 1
    else:
        y_range = y_max - y_min or 1
        y_min_plot = y_min - y_range * 0.05
        y_max_plot = y_max + y_range * 0.15

    n_groups = len(labels)
    n_series = len(series)
    group_width = cw / n_groups
    bar_width = group_width * 0.7 / max(n_series, 1)
    gap = group_width * 0.15

    def sy(v):
        return mt + ch - ((v - y_min_plot) / (y_max_plot - y_min_plot)) * ch

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    for i in range(6):
        y_tick = y_min_plot + (y_max_plot - y_min_plot) * i / 5
        py = sy(y_tick)
        svg.append(f'<line x1="{ml}" y1="{py:.1f}" x2="{ml + cw}" y2="{py:.1f}" stroke="#e9ecef" stroke-width="1"/>')
        svg.append(f'<text x="{ml - 8}" y="{py + 4:.1f}" text-anchor="end" font-size="11" fill="#6c757d">{y_tick:{value_format}}</text>')

    for gi, label in enumerate(labels):
        gx = ml + gi * group_width + gap
        for si, (name, vals) in enumerate(series.items()):
            color = colors[si % len(colors)]
            bx = gx + si * bar_width
            val = vals[gi]
            by = sy(val)
            bh = mt + ch - by
            svg.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_width - 1:.1f}" height="{max(0, bh):.1f}" fill="{color}" rx="2"/>')
            svg.append(f'<text x="{bx + bar_width / 2:.1f}" y="{by - 4:.1f}" text-anchor="middle" font-size="9" fill="#1a1a2e">{val:{value_format}}</text>')
        lx = gx + n_series * bar_width / 2
        svg.append(f'<text x="{lx:.1f}" y="{mt + ch + 14}" text-anchor="start" font-size="10" fill="#6c757d" transform="rotate(35, {lx:.1f}, {mt + ch + 14})">{label}</text>')

    if title:
        svg.append(f'<text x="{width / 2}" y="22" text-anchor="middle" font-size="14" font-weight="600" fill="#1a1a2e">{title}</text>')

    if n_series > 1:
        lx = ml + cw - len(series) * 110
        for si, name in enumerate(series):
            color = colors[si % len(colors)]
            svg.append(f'<rect x="{lx + si * 110}" y="{mt + ch + 55}" width="12" height="12" rx="2" fill="{color}"/>')
            svg.append(f'<text x="{lx + si * 110 + 16}" y="{mt + ch + 66}" font-size="11" fill="#1a1a2e">{name}</text>')

    svg.append("</svg>")
    return "\n".join(svg)


def _make_plddt_sparkline(values: list[float], width: int = 400, height: int = 50) -> str:
    """pLDDT sparkline with AlphaFold-style coloring."""
    if not values or len(values) < 2:
        return ""

    pad = 4
    cw = width - 2 * pad
    ch = height - 2 * pad
    seg_w = cw / len(values)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
    ]

    for i, v in enumerate(values):
        x = pad + i * seg_w
        bar_h = (v / 100) * ch
        y = pad + ch - bar_h

        if v >= 90:
            color = "#0053d6"
        elif v >= 70:
            color = "#65cbf3"
        elif v >= 50:
            color = "#ffdb13"
        else:
            color = "#ff7d45"

        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(seg_w, 1):.1f}" height="{bar_h:.1f}" fill="{color}"/>')

    ref_y = pad + ch - (70 / 100) * ch
    svg.append(f'<line x1="{pad}" y1="{ref_y:.1f}" x2="{pad + cw}" y2="{ref_y:.1f}" stroke="#adb5bd" stroke-width="0.5" stroke-dasharray="3,2"/>')

    svg.append("</svg>")
    return "\n".join(svg)


def _outputs_to_pdb(outputs, sequence: str) -> str:
    """Convert ESMFold outputs to PDB format string."""
    import numpy as np

    pos = outputs.positions[0]
    if pos.dim() == 4:
        pos = pos[-1]
    positions = pos.cpu().numpy()
    atom_names = ["N", "CA", "C", "O"]
    aa_3letter = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    }

    pdb_lines = []
    atom_idx = 1
    for res_idx, aa in enumerate(sequence):
        res_name = aa_3letter.get(aa, "UNK")
        for atom_i, atom_name in enumerate(atom_names):
            if atom_i >= positions.shape[1]:
                break
            x, y, z = positions[res_idx, atom_i]
            if any(math.isnan(c) for c in (x, y, z)):
                continue
            pdb_lines.append(
                f"ATOM  {atom_idx:5d}  {atom_name:<3s} {res_name} A{res_idx + 1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]:>2s}"
            )
            atom_idx += 1
    pdb_lines.append("END")
    return "\n".join(pdb_lines)


# ------------------------------------------------------------------
# Task 1: Load gene set
# ------------------------------------------------------------------

@cpu_env.task()
async def load_genes(
    gene_set: str = "insulin",
    custom_json: str = "",
) -> flyte.io.Dir:
    """Load a set of homologous genes from different species."""
    if custom_json:
        data = json.loads(custom_json)
    elif gene_set in GENE_SETS:
        data = GENE_SETS[gene_set]
    else:
        available = ", ".join(GENE_SETS.keys())
        raise ValueError(f"Unknown gene set '{gene_set}'. Available: {available}")

    log.info(f"Loaded gene set: {data['gene_name']} - {len(data['sequences'])} species")

    out_dir = tempfile.mkdtemp(prefix="gene_compare_")
    with open(os.path.join(out_dir, "genes.json"), "w") as f:
        json.dump(data, f)

    return await flyte.io.Dir.from_local(out_dir)


# ------------------------------------------------------------------
# Task 2: Score sequences with Carbon
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def score_sequences(
    genes_dir: flyte.io.Dir,
    model_name: str = "HuggingFaceBio/Carbon-3B",
) -> str:
    """Score each species' gene with Carbon-3B genomic language model.

    Returns per-species log-likelihood scores and sequence metadata.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"Loading Carbon model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    genes_path = await genes_dir.download()
    with open(os.path.join(genes_path, "genes.json")) as f:
        data = json.load(f)

    species_names = list(data["sequences"].keys())
    n = len(species_names)

    scores = {}
    for i, species in enumerate(species_names):
        await flyte.report.replace.aio(_wrap_report(
            f"<h2>Carbon Scoring</h2>"
            f"<p>Scoring {species} ({i + 1}/{n})...</p>"
        ), do_flush=True)

        dna = data["sequences"][species]["dna"]
        prompt = f"<dna>{dna}"
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

        with torch.no_grad():
            output = model(**inputs, labels=inputs["input_ids"])
            loss = output.loss.item()
            ll = -loss * inputs["input_ids"].shape[1]

        protein = _translate(dna)
        scores[species] = {
            "log_likelihood": round(ll, 4),
            "loss": round(loss, 4),
            "gc_content": round(_gc_content(dna), 4),
            "length": len(dna),
            "protein": protein,
            "protein_length": len(protein),
            "common_name": data["sequences"][species]["common_name"],
        }
        log.info(f"  {species} ({data['sequences'][species]['common_name']}): LL={ll:.2f}, GC={_gc_content(dna):.1%}")

    # Report
    html_parts = [
        f"<h2>{data['gene_name']} - Carbon Scoring</h2>",
        f'<div class="note">{data["description"]}</div>',
        '<div class="stat-grid">',
        f'<div class="stat"><div class="value">{n}</div><div class="label">Species</div></div>',
        f'<div class="stat"><div class="value">{data["gene_name"]}</div><div class="label">Gene</div></div>',
        f'<div class="stat"><div class="value">{model_name.split("/")[-1]}</div><div class="label">Model</div></div>',
        "</div>",
    ]

    html_parts.append(
        "<table><tr><th>Species</th><th>Scientific Name</th><th>DNA Length</th>"
        "<th>GC%</th><th>Protein Length</th><th>Carbon LL</th></tr>"
    )
    for species in species_names:
        s = scores[species]
        html_parts.append(
            f'<tr><td><b>{species}</b></td><td><i>{s["common_name"]}</i></td>'
            f'<td>{s["length"]}bp</td><td>{s["gc_content"]:.1%}</td>'
            f'<td>{s["protein_length"]}aa</td>'
            f'<td>{s["log_likelihood"]:.2f}</td></tr>'
        )
    html_parts.append("</table>")

    html_parts.append('<div class="chart-container">')
    html_parts.append(_make_bar_chart(
        species_names,
        {"Log-Likelihood": [scores[s]["log_likelihood"] for s in species_names]},
        title="Carbon Log-Likelihood per Species",
        value_format=".1f",
    ))
    html_parts.append("</div>")

    await flyte.report.replace.aio(_wrap_report("\n".join(html_parts)), do_flush=True)

    result = {
        "gene_name": data["gene_name"],
        "description": data["description"],
        "species": species_names,
        "scores": scores,
    }
    return json.dumps(result)


# ------------------------------------------------------------------
# Task 3: Align sequences and compute similarity
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def align_and_compare(
    scores_json: str,
    genes_dir: flyte.io.Dir,
) -> str:
    """Align sequences with Needleman-Wunsch and compute pairwise identity.

    Translates DNA to protein, builds DNA and protein identity matrices,
    and generates phylogenetic trees from sequence divergence.
    """
    scores_data = json.loads(scores_json)
    species_names = scores_data["species"]
    scores = scores_data["scores"]
    gene_name = scores_data["gene_name"]
    n = len(species_names)

    genes_path = await genes_dir.download()
    with open(os.path.join(genes_path, "genes.json")) as f:
        data = json.load(f)

    await flyte.report.replace.aio(_wrap_report(
        f"<h2>{gene_name} - Sequence Alignment</h2>"
        f"<p>Aligning {n} species with Needleman-Wunsch...</p>"
    ), do_flush=True)

    # Pairwise DNA identity matrix
    identity_matrix = []
    for sp1 in species_names:
        row = []
        for sp2 in species_names:
            dna1 = data["sequences"][sp1]["dna"]
            dna2 = data["sequences"][sp2]["dna"]
            identity = _sequence_identity(dna1, dna2)
            row.append(round(identity, 4))
        identity_matrix.append(row)

    # Pairwise protein identity matrix
    protein_matrix = []
    for sp1 in species_names:
        row = []
        for sp2 in species_names:
            identity = _sequence_identity(scores[sp1]["protein"], scores[sp2]["protein"])
            row.append(round(identity, 4))
        protein_matrix.append(row)

    # Average pairwise identities (exclude diagonal)
    dna_pairs = [identity_matrix[i][j] for i in range(n) for j in range(i + 1, n)]
    prot_pairs = [protein_matrix[i][j] for i in range(n) for j in range(i + 1, n)]
    avg_dna = sum(dna_pairs) / len(dna_pairs) if dna_pairs else 0
    avg_prot = sum(prot_pairs) / len(prot_pairs) if prot_pairs else 0

    # Most/least similar pair
    best_pair = max(range(len(dna_pairs)), key=lambda k: dna_pairs[k])
    worst_pair = min(range(len(dna_pairs)), key=lambda k: dna_pairs[k])
    pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
    best_sp = f"{species_names[pair_indices[best_pair][0]]}-{species_names[pair_indices[best_pair][1]]}"
    worst_sp = f"{species_names[pair_indices[worst_pair][0]]}-{species_names[pair_indices[worst_pair][1]]}"

    # Report
    html_parts = [
        f"<h2>{gene_name} - Sequence Alignment</h2>",
        f'<div class="note">Pairwise alignment using Needleman-Wunsch (match=2, mismatch=-1, gap=-2). '
        f"Identity is computed as matches / aligned length from the optimal global alignment.</div>",
        '<div class="stat-grid">',
        f'<div class="stat"><div class="value">{n}</div><div class="label">Species Aligned</div></div>',
        f'<div class="stat"><div class="value">{n * (n - 1) // 2}</div><div class="label">Pairwise Alignments</div></div>',
        f'<div class="stat"><div class="value">{avg_dna:.0%}</div><div class="label">Avg DNA Identity</div></div>',
        f'<div class="stat"><div class="value">{avg_prot:.0%}</div><div class="label">Avg Protein Identity</div></div>',
        f'<div class="stat"><div class="value">{best_sp}</div><div class="label">Most Similar</div></div>',
        f'<div class="stat"><div class="value">{worst_sp}</div><div class="label">Most Divergent</div></div>',
        "</div>",
    ]

    # DNA identity heatmap
    html_parts.append('<div class="chart-container">')
    html_parts.append(_make_heatmap(
        identity_matrix, species_names, species_names,
        title="Pairwise DNA Sequence Identity (%)",
        value_format=".0%",
    ))
    html_parts.append("</div>")

    # Protein identity heatmap
    html_parts.append('<div class="chart-container">')
    html_parts.append(_make_heatmap(
        protein_matrix, species_names, species_names,
        title="Pairwise Protein Sequence Identity (%)",
        value_format=".0%",
        color_scale="green",
    ))
    html_parts.append("</div>")

    # DNA phylogenetic tree
    html_parts.append('<div class="chart-container">')
    html_parts.append(_make_dendrogram(
        species_names, identity_matrix,
        title=f"{gene_name} - Phylogenetic Tree (DNA Identity)",
    ))
    html_parts.append("</div>")

    # Protein phylogenetic tree
    html_parts.append('<div class="chart-container">')
    html_parts.append(_make_dendrogram(
        species_names, protein_matrix,
        title=f"{gene_name} - Phylogenetic Tree (Protein Identity)",
        color="#059669",
    ))
    html_parts.append("</div>")

    # DNA vs Protein conservation comparison
    html_parts.append('<div class="chart-container">')
    html_parts.append(_make_bar_chart(
        species_names,
        {
            "DNA vs Human": [identity_matrix[0][j] for j in range(n)],
            "Protein vs Human": [protein_matrix[0][j] for j in range(n)],
        },
        title=f"Conservation vs {species_names[0]} (DNA and Protein)",
        value_format=".0%",
    ))
    html_parts.append("</div>")

    await flyte.report.replace.aio(_wrap_report("\n".join(html_parts)), do_flush=True)

    result = {
        "gene_name": gene_name,
        "description": scores_data["description"],
        "species": species_names,
        "scores": scores,
        "dna_identity_matrix": identity_matrix,
        "protein_identity_matrix": protein_matrix,
    }
    return json.dumps(result)


# ------------------------------------------------------------------
# Task 4: Fold proteins with ESMFold
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def fold_proteins(
    comparison_json: str,
    max_length: int = 400,
) -> str:
    """Fold each species' translated protein with ESMFold for 3D comparison.

    Returns PDB strings and pLDDT confidence scores for each species.
    """
    import torch
    import numpy as np
    from transformers import AutoTokenizer, EsmForProteinFolding

    comparison = json.loads(comparison_json)
    species_names = comparison["species"]
    scores = comparison["scores"]

    log.info("Loading ESMFold model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.to(device)
    model.eval()

    structure_data = {}
    n = len(species_names)

    for idx, species in enumerate(species_names):
        protein = scores[species]["protein"]

        if len(protein) > max_length:
            log.info(f"Skipping {species} ({len(protein)} aa > {max_length} max)")
            continue

        log.info(f"ESMFold [{idx + 1}/{n}]: {species} ({len(protein)} aa)")
        await flyte.report.replace.aio(_wrap_report(
            f"<h2>ESMFold - 3D Structure Prediction</h2>"
            f"<p>Folding {species} ({idx + 1}/{n}): {len(protein)} residues...</p>"
        ), do_flush=True)

        inputs = tokenizer(protein, return_tensors="pt", add_special_tokens=False).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        pdb_str = _outputs_to_pdb(outputs, protein)

        plddt_raw = outputs.plddt[0].cpu().numpy()
        if plddt_raw.ndim == 2:
            plddt_raw = plddt_raw[-1]
        plddt = plddt_raw.flatten()[:len(protein)]
        if plddt.max() <= 1.0:
            plddt = plddt * 100
        plddt_mean = float(np.mean(plddt))

        structure_data[species] = {
            "pdb_str": pdb_str,
            "plddt_mean": round(plddt_mean, 1),
            "plddt_per_residue": [round(float(v), 1) for v in plddt[:len(protein)]],
            "protein_length": len(protein),
        }
        log.info(f"  → mean pLDDT: {plddt_mean:.1f}")

    # Report with 3D viewers
    n_folded = len(structure_data)
    avg_plddt = sum(d["plddt_mean"] for d in structure_data.values()) / n_folded if n_folded else 0

    threeDmol_script = '<script src="https://3dmol.csb.pitt.edu/build/3Dmol-min.js"></script>'

    stats_html = f"""
    <h2>ESMFold - Cross-Species Structure Comparison</h2>
    <div class="note">
      <b>ESMFold</b> predicts 3D structure directly from amino acid sequence.
      Comparing structures across species reveals which parts of the protein are
      structurally conserved (functional core) vs divergent (surface loops, species-specific adaptations).
    </div>
    <div class="stat-grid">
      <div class="stat"><div class="value">{n_folded}</div><div class="label">Structures</div></div>
      <div class="stat"><div class="value">{avg_plddt:.1f}</div><div class="label">Avg pLDDT</div></div>
      <div class="stat"><div class="value">{comparison['gene_name']}</div><div class="label">Gene</div></div>
    </div>
    """

    viewers_html = '<div class="structure-grid">'
    for species, sdata in structure_data.items():
        plddt_val = sdata["plddt_mean"]
        common = scores[species]["common_name"]

        if plddt_val >= 90:
            badge = '<span class="badge badge-success">Very High</span>'
        elif plddt_val >= 70:
            badge = '<span class="badge badge-info">Confident</span>'
        elif plddt_val >= 50:
            badge = '<span class="badge badge-warning">Low</span>'
        else:
            badge = '<span class="badge badge-danger">Disordered</span>'

        plddt_sparkline = _make_plddt_sparkline(sdata["plddt_per_residue"], width=300)
        pdb_escaped = sdata["pdb_str"].replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
        viewer_id = f"viewer_{hash(species) & 0xFFFFFF:06x}"

        viewers_html += f"""
        <div class="card" style="margin:0;">
          <h3 style="margin-top:0;">{species}
            <span style="font-size:0.7em;color:#6c757d;">({sdata['protein_length']} aa)</span>
            {badge}
          </h3>
          <p style="font-size:0.85em;color:#6c757d;margin:2px 0 8px;"><i>{common}</i></p>
          <div id="{viewer_id}" style="width:100%;max-width:320px;height:280px;border:1px solid #dbeafe;border-radius:8px;position:relative;"></div>
          <div style="margin-top:8px;">
            <b>Mean pLDDT:</b> {plddt_val:.1f} / 100
            <div style="margin-top:4px;">{plddt_sparkline}</div>
            <div style="font-size:0.75em;color:#9ca3af;margin-top:2px;">
              <span style="color:#0053d6;">&block; &gt;90</span>
              <span style="color:#65cbf3;">&block; 70-90</span>
              <span style="color:#ffdb13;">&block; 50-70</span>
              <span style="color:#ff7d45;">&block; &lt;50</span>
            </div>
          </div>
        </div>
        <script>
        (function() {{
          var pdb = `{pdb_escaped}`;
          function initViewer() {{
            if (typeof $3Dmol === 'undefined') {{ setTimeout(initViewer, 200); return; }}
            var viewer = $3Dmol.createViewer(document.getElementById("{viewer_id}"), {{backgroundColor: "white"}});
            viewer.addModel(pdb, "pdb");
            viewer.setStyle({{}}, {{cartoon: {{color: "spectrum"}}}});
            viewer.zoomTo();
            viewer.render();
            viewer.spin("y", 1);
          }}
          initViewer();
        }})();
        </script>
        """

    viewers_html += "</div>"

    # pLDDT comparison bar chart
    plddt_chart = _make_bar_chart(
        list(structure_data.keys()),
        {"Mean pLDDT": [d["plddt_mean"] for d in structure_data.values()]},
        title="Structure Confidence Comparison (pLDDT)",
        value_format=".1f",
        colors=["#0053d6"],
    )

    report_html = f"""
    {threeDmol_script}
    {stats_html}
    {viewers_html}
    <div class="chart-container">{plddt_chart}</div>
    """

    await flyte.report.replace.aio(_wrap_report(report_html), do_flush=True)

    return json.dumps(structure_data)


# ------------------------------------------------------------------
# Task 5: Generate summary
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def generate_summary(
    comparison_json: str,
    structures_json: str,
) -> str:
    """Generate comprehensive cross-species summary."""
    comparison = json.loads(comparison_json)
    structures = json.loads(structures_json)

    species = comparison["species"]
    scores = comparison["scores"]
    gene_name = comparison["gene_name"]
    dna_matrix = comparison["dna_identity_matrix"]
    protein_matrix = comparison["protein_identity_matrix"]

    html_parts = [
        f"<h2>{gene_name} - Cross-Species Evolution Summary</h2>",
        f'<div class="note">{comparison["description"]}</div>',
    ]

    # Key metrics
    # Average pairwise identity (exclude diagonal)
    n = len(species)
    dna_pairs = [dna_matrix[i][j] for i in range(n) for j in range(i + 1, n)]
    protein_pairs = [protein_matrix[i][j] for i in range(n) for j in range(i + 1, n)]
    avg_dna_id = sum(dna_pairs) / len(dna_pairs) if dna_pairs else 0
    avg_protein_id = sum(protein_pairs) / len(protein_pairs) if protein_pairs else 0
    avg_plddt = sum(d["plddt_mean"] for d in structures.values()) / len(structures) if structures else 0

    html_parts.append('<div class="stat-grid">')
    html_parts.append(f'<div class="stat"><div class="value">{n}</div><div class="label">Species</div></div>')
    html_parts.append(f'<div class="stat"><div class="value">{avg_dna_id:.0%}</div><div class="label">Avg DNA Identity</div></div>')
    html_parts.append(f'<div class="stat"><div class="value">{avg_protein_id:.0%}</div><div class="label">Avg Protein Identity</div></div>')
    html_parts.append(f'<div class="stat"><div class="value">{avg_plddt:.1f}</div><div class="label">Avg pLDDT</div></div>')
    html_parts.append(f'<div class="stat"><div class="value">{len(structures)}</div><div class="label">Structures Folded</div></div>')
    html_parts.append("</div>")

    # Full comparison table
    html_parts.append("<h3>Per-Species Detail</h3>")
    html_parts.append(
        "<table><tr><th>Species</th><th>Scientific Name</th><th>DNA (bp)</th>"
        "<th>Protein (aa)</th><th>GC%</th><th>Carbon LL</th><th>pLDDT</th></tr>"
    )
    for sp in species:
        s = scores[sp]
        plddt = structures.get(sp, {}).get("plddt_mean", "N/A")
        plddt_str = f"{plddt:.1f}" if isinstance(plddt, float) else plddt
        html_parts.append(
            f'<tr><td><b>{sp}</b></td><td><i>{s["common_name"]}</i></td>'
            f'<td>{s["length"]}</td><td>{s["protein_length"]}</td>'
            f'<td>{s["gc_content"]:.1%}</td><td>{s["log_likelihood"]:.2f}</td>'
            f'<td>{plddt_str}</td></tr>'
        )
    html_parts.append("</table>")

    # GC content comparison
    html_parts.append('<div class="chart-container">')
    html_parts.append(_make_bar_chart(
        species,
        {"GC Content": [scores[s]["gc_content"] for s in species]},
        title="GC Content Across Species",
        value_format=".2f",
    ))
    html_parts.append("</div>")

    # DNA phylogenetic tree
    html_parts.append("<h3>Phylogenetic Relationships</h3>")
    html_parts.append(
        '<div class="note">'
        "Trees built from pairwise sequence identity using UPGMA clustering. "
        "Species that diverged more recently cluster together. DNA and protein trees "
        "may differ when synonymous mutations dominate."
        "</div>"
    )

    html_parts.append('<div class="chart-container">')
    html_parts.append(_make_dendrogram(
        species, dna_matrix,
        title=f"{gene_name} - DNA Phylogenetic Tree",
    ))
    html_parts.append("</div>")

    html_parts.append('<div class="chart-container">')
    html_parts.append(_make_dendrogram(
        species, protein_matrix,
        title=f"{gene_name} - Protein Phylogenetic Tree",
        color="#059669",
    ))
    html_parts.append("</div>")

    await flyte.report.replace.aio(_wrap_report("\n".join(html_parts)), do_flush=True)

    summary = {
        "gene_name": gene_name,
        "n_species": n,
        "avg_dna_identity": round(avg_dna_id, 4),
        "avg_protein_identity": round(avg_protein_id, 4),
        "avg_plddt": round(avg_plddt, 1),
        "n_structures": len(structures),
    }
    return json.dumps(summary)


# ------------------------------------------------------------------
# Pipeline orchestrator
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(
    gene_set: str = "insulin",
    model_name: str = "HuggingFaceBio/Carbon-3B",
    custom_json: str = "",
) -> tuple[str, str]:
    """
    End-to-end cross-species gene comparison pipeline.

    Returns (comparison JSON, structures JSON).

    1. Load homologous gene sequences across species
    2. Score with Carbon genomic language model
    3. Align sequences and compute pairwise similarity
    4. Fold translated proteins with ESMFold
    5. Generate comprehensive summary with phylogenetic trees
    """
    log.info(f"Starting cross-species gene comparison pipeline (gene_set={gene_set})...")

    def _pipeline_progress(step: int, label: str) -> str:
        steps = [
            "Load Genes",
            "Carbon Scoring",
            "Sequence Alignment",
            "ESMFold Structures",
            "Generate Summary",
        ]
        dots = ""
        for i, s in enumerate(steps):
            if i + 1 < step:
                icon = '<span style="color:#2563eb;">&#10003;</span>'
            elif i + 1 == step:
                icon = '<span style="color:#2563eb;">&#9679;</span>'
            else:
                icon = '<span style="color:#adb5bd;">&#9675;</span>'
            dots += f"<span style='margin:0 8px;'>{icon} {s}</span>"
        return f"""
        <h2>Cross-Species Gene Comparison</h2>
        <div class="card" style="text-align:center;">{dots}</div>
        <p>{label}</p>
        """

    # Stage 1
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(1, "Loading homologous gene sequences...")),
        do_flush=True,
    )
    genes_dir = await load_genes(gene_set=gene_set, custom_json=custom_json)

    # Stage 2
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(2, "Scoring sequences with Carbon...")),
        do_flush=True,
    )
    scores_json = await score_sequences(genes_dir=genes_dir, model_name=model_name)

    # Stage 3
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(3, "Aligning sequences with Needleman-Wunsch...")),
        do_flush=True,
    )
    comparison_json = await align_and_compare(scores_json=scores_json, genes_dir=genes_dir)

    # Stage 4
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(4, "Folding proteins with ESMFold...")),
        do_flush=True,
    )
    structures_json = await fold_proteins(comparison_json=comparison_json)

    # Stage 5
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(5, "Generating summary report...")),
        do_flush=True,
    )
    summary_json = await generate_summary(
        comparison_json=comparison_json,
        structures_json=structures_json,
    )

    # Final report
    summary = json.loads(summary_json)
    comparison = json.loads(comparison_json)

    final_html = f"""
    <h2>Pipeline Complete</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{summary['gene_name']}</div><div class="label">Gene</div></div>
      <div class="stat"><div class="value">{summary['n_species']}</div><div class="label">Species</div></div>
      <div class="stat"><div class="value">{summary['avg_dna_identity']:.0%}</div><div class="label">Avg DNA Identity</div></div>
      <div class="stat"><div class="value">{summary['avg_protein_identity']:.0%}</div><div class="label">Avg Protein Identity</div></div>
      <div class="stat"><div class="value">{summary['avg_plddt']:.1f}</div><div class="label">Avg pLDDT</div></div>
      <div class="stat"><div class="value">{summary['n_structures']}</div><div class="label">3D Structures</div></div>
    </div>
    <div class="card">
      <b>Gene:</b> {summary['gene_name']} |
      <b>Species:</b> {', '.join(comparison['species'])} |
      <b>Model:</b> {model_name}
    </div>
    <div class="note">
      All 4 pipeline stages completed. View individual task reports for DNA/protein
      identity heatmaps, phylogenetic trees, interactive 3D protein structures with
      pLDDT confidence, Carbon log-likelihood scores, and evolutionary analysis.
    </div>
    """

    await flyte.report.replace.aio(_wrap_report(final_html), do_flush=True)
    log.info("Pipeline complete.")
    return comparison_json, structures_json
