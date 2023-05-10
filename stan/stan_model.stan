functions {
  /**
   * Return the matrix corresponding to the fast Fourier
   * transform of Z after it is padded with zeros to size
   * N by M
   * When N by M is larger than the dimensions of Z,
   * this computes an oversampled FFT.
   *
   * @param Z matrix of values
   * @param N number of rows desired (must be >= rows(Z))
   * @param M number of columns desired (must be >= cols(Z))
   * @return the FFT of Z padded with zeros
   */
  complex_matrix fft2(complex_matrix Z, int N, int M) {
    int r = rows(Z);
    int c = cols(Z);
    if (r > N) {
      reject("N must be at least rows(Z)");
    }
    if (c > M) {
      reject("M must be at least cols(Z)");
    }
    
    complex_matrix[N, M] pad = rep_matrix(0, N, M);
    pad[1 : r, 1 : c] = Z;
    
    return fft2(pad);
  }
}

data {
  int<lower=0> N; // image dimension
  matrix<lower=0, upper=1>[N, N] S; // registration image
  int<lower=0, upper=N> d; // separation between sample and registration image
  int<lower=N> M1; // rows of padded matrices
  int<lower=2 * N + d> M2; // cols of padded matrices

  real<lower=0> N_p; // avg number of photons per pixel

  array[M1, M2] int<lower=0> Y_tilde; // observed number of photons
}

transformed data {
  matrix[d, N] separation = rep_matrix(0, d, N);
}

parameters {
  matrix<lower=0, upper=1>[N, N] X;
}

model {
  
  // likelihood
  matrix[N, 2 * N + d] X0S = append_col(X, append_col(separation, S));
  // signal - squared magnitude of the (oversampled) FFT
  matrix[M1, M2] Y = abs(fft2(X0S, M1, M2)) .^ 2;
  
  real N_p_over_Y_bar = N_p / mean(Y);
  matrix[M1, M2] lambda = N_p_over_Y_bar * Y;

  for (m1 in 1 : M1) {
    for (m2 in 1 : M2) {
      Y_tilde[m1, m2] ~ poisson(lambda[m1, m2]);
    }
  }
}