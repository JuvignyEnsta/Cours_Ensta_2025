#include <algorithm>
#include <cassert>
#include <iostream>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "ProdMatMat.hpp"

namespace {
void prodSubBlocks(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
                   const Matrix& A, const Matrix& B, Matrix& C) {
    const int iEnd = std::min(A.nbRows, iRowBlkA + szBlock);
    const int kEnd = std::min(A.nbCols, iColBlkA + szBlock);
    const int jEnd = std::min(B.nbCols, iColBlkB + szBlock);
    
    // Optimisation mémoire avec pré-chargement des blocs
    for (int k = iColBlkA; k < kEnd; ++k) {
        for (int j = iColBlkB; j < jEnd; ++j) {
            const double b_kj = B(k, j); // Pré-chargement car accès multiple à B
            for (int i = iRowBlkA; i < iEnd; ++i) {
                C(i, j) += A(i, k) * b_kj;
            }
        }
    }
}

// Taille de bloc adaptée à la hiérarchie mémoire de mon pd
constexpr int szBlock = 128; // Optimisé pour L2 cache (1.5MB/core)
}  // namespace

Matrix operator*(const Matrix& A, const Matrix& B) {
  Matrix C(A.nbRows, B.nbCols, 0.0);

  #pragma omp parallel for  // Charge dynamique pour déséquilibres
  for (int iRowBlkA = 0; iRowBlkA < A.nbRows; iRowBlkA += szBlock) {
    for (int iColBlkA = 0; iColBlkA < A.nbCols; iColBlkA += szBlock) {
      for (int iColBlkB = 0; iColBlkB < B.nbCols; iColBlkB += szBlock) {
        prodSubBlocks(iRowBlkA, iColBlkB, iColBlkA, szBlock, A, B, C);
      }
    }
  }
  
  return C;
}