"""Sudoku görüntü işleme pipeline'ı.

Modüller ve görevleri:
  io_utils    – dosyadan okuma / dosyaya yazma (başka hiçbir şey)
  preprocess  – Stage 1: küçültme, gri ton, gürültü azaltma, aydınlatma düzeltme
  corners     – Stage 2a: Harris R haritası, aday köşeler, koordinat dönüşümü
  visualize   – debug görselleri (sadece çizim; hesaplama yapmaz)
  rectify, cells, ocr, validate – ileriki fazlar
"""
