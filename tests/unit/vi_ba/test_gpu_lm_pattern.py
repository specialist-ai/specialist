"""Tests for the cuDSS sparsity-pattern guard: reusing a factorization plan
across a pattern change writes values into the wrong slots."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from specialist.vi_ba.gpu_lm import same_pattern


def test_same_values_same_pattern() -> None:
  a = sp.csr_matrix(np.array([[1.0, 0.0], [2.0, 3.0]]))
  b = sp.csr_matrix(np.array([[9.0, 0.0], [8.0, 7.0]]))
  assert same_pattern((a.indptr, a.indices), b)


def test_changed_nnz_is_a_different_pattern() -> None:
  a = sp.csr_matrix(np.array([[1.0, 0.0], [2.0, 3.0]]))
  b = sp.csr_matrix(np.array([[1.0, 4.0], [2.0, 3.0]]))
  assert not same_pattern((a.indptr, a.indices), b)


def test_same_nnz_different_slots_is_a_different_pattern() -> None:
  a = sp.csr_matrix(np.array([[1.0, 0.0], [2.0, 3.0]]))
  b = sp.csr_matrix(np.array([[0.0, 1.0], [2.0, 3.0]]))
  assert not same_pattern((a.indptr, a.indices), b)
