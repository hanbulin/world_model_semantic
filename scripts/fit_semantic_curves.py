import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def logistic(x, a, b, c, d):
  return a / (1.0 + np.exp(-b * (x - c))) + d


def main():
  root = pathlib.Path(__file__).resolve().parents[1]
  ref_path = root / 'configs' / 'semantic_fit_reference.json'
  with ref_path.open('r', encoding='utf-8') as f:
    ref = json.load(f)

  x = np.array(ref['snr_db_points'], dtype=float)
  y_mid = np.array(ref['rayleigh_sentence_similarity_points'], dtype=float)
  y_awgn = np.array(ref['awgn_sentence_similarity_points'], dtype=float)

  bounds = ([0.0, 0.0, -10.0, 0.0], [1.0, 5.0, 20.0, 1.0])
  mid, _ = curve_fit(
      logistic, x, y_mid, p0=[0.35, 0.35, 3.5, 0.30],
      bounds=bounds, maxfev=100000)
  awgn, _ = curve_fit(
      logistic, x, y_awgn, p0=[0.95, 0.4, -2.0, 0.0],
      bounds=bounds, maxfev=100000)

  low = np.array([
      mid[0] * 0.85,
      mid[1] * 0.85,
      mid[2] + 2.0,
      mid[3] * 0.90,
  ])
  high = np.array([
      min(0.98 - mid[3] * 1.05, mid[0] * 1.05),
      mid[1] * 1.15,
      mid[2] - 1.0,
      min(0.5, mid[3] * 1.05),
  ])

  x_dense = np.linspace(0, 18, 400)
  plt.figure(figsize=(8, 5))
  plt.scatter(x, y_mid, color='black', label='DeepSC Fig.7(b) points')
  plt.plot(x_dense, logistic(x_dense, *mid), color='black', label='Mid fit')
  plt.plot(x_dense, logistic(x_dense, *low), '--', color='tab:blue', label='Low derived')
  plt.plot(x_dense, logistic(x_dense, *high), '--', color='tab:red', label='High derived')
  plt.plot(x_dense, logistic(x_dense, *awgn), ':', color='tab:green', label='AWGN reference')
  plt.xlabel('SNR (dB)')
  plt.ylabel('Sentence similarity')
  plt.ylim(0, 1.02)
  plt.grid(True, alpha=0.3)
  plt.legend()
  out_png = root / 'outputs' / 'semantic_similarity_fits.png'
  plt.tight_layout()
  plt.savefig(out_png, dpi=200)

  out_json = root / 'outputs' / 'semantic_fit_results.json'
  result = {
      'low': {'A': round(float(low[0]), 4), 'B': round(float(low[1]), 4),
              'C': round(float(low[2]), 4), 'D': round(float(low[3]), 4)},
      'mid': {'A': round(float(mid[0]), 4), 'B': round(float(mid[1]), 4),
              'C': round(float(mid[2]), 4), 'D': round(float(mid[3]), 4)},
      'high': {'A': round(float(high[0]), 4), 'B': round(float(high[1]), 4),
               'C': round(float(high[2]), 4), 'D': round(float(high[3]), 4)},
      'awgn_reference': {'A': round(float(awgn[0]), 4), 'B': round(float(awgn[1]), 4),
                         'C': round(float(awgn[2]), 4), 'D': round(float(awgn[3]), 4)},
  }
  out_json.write_text(json.dumps(result, indent=2), encoding='utf-8')
  print(out_png)
  print(out_json)


if __name__ == '__main__':
  main()
