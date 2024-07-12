
import scanpy as sc
import pkg_resources
import lzma
import pickle


def preendocrine_alpha() -> sc.AnnData:
    try:
        ad = sc.read_h5ad(pkg_resources.resource_filename(__name__, '/d/pre-endocrine_alpha.h5ad'))
    except FileNotFoundError:
        # Deserialize (unpickle) and decompress the AnnData object
        with lzma.open(pkg_resources.resource_filename(__name__, '/d/pre-endocrine_alpha.pickle.xz'), 'rb') as f:
            ad = pickle.load(f)
        sc.write(pkg_resources.resource_filename(__name__, '/d/pre-endocrine_alpha.h5ad'), ad)
    return ad


def preendocrine_beta() -> sc.AnnData:
    try:
        ad = sc.read_h5ad(pkg_resources.resource_filename(__name__, '/d/pre-endocrine_beta.h5ad'))
    except FileNotFoundError:
        # Deserialize (unpickle) and decompress the AnnData object
        with lzma.open(pkg_resources.resource_filename(__name__, '/d/pre-endocrine_beta.pickle.xz'), 'rb') as f:
            ad = pickle.load(f)
        sc.write(pkg_resources.resource_filename(__name__, '/d/pre-endocrine_beta.h5ad'), ad)
    return ad


def erythrocytes() -> sc.AnnData:
    try:
        ad = sc.read_h5ad(pkg_resources.resource_filename(__name__, '/d/erythrocytes.h5ad'))
    except FileNotFoundError:
        # Deserialize (unpickle) and decompress the AnnData object
        with lzma.open(pkg_resources.resource_filename(__name__, '/d/erythrocytes.pickle.xz'), 'rb') as f:
            ad = pickle.load(f)
        sc.write(pkg_resources.resource_filename(__name__, '/d/erythrocytes.h5ad'), ad)
    return ad
