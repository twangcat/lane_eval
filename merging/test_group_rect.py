import numpy as np
import cv2
import group_rectangles_with_aux
import random

num_trials = 10000
num_rectangles = 500
arena_size = 100.0
square_size = 10.0
gt = 1
eps = 0.8
use_aux = True

for t in range(num_trials):
  opencv_gr = []
  my_gr = []
  for r in range(num_rectangles):
    e = np.array([
        random.random() * arena_size, random.random() * arena_size,
        random.random() * square_size, random.random() * square_size])
    ec = np.concatenate([e, np.array([random.random(), random.random()])])
    opencv_gr.append(e)
    my_gr.append(ec if use_aux else e)

  output1, scores1 = cv2.groupRectangles(opencv_gr, gt, eps)
  output2, aux2, scores2 = group_rectangles_with_aux.execute(my_gr, gt, eps)
  if len(output1) == 0:
    assert len(output2) == 0
  else:
    try:
      assert (output1 == output2).all()
    except:
      print 'opencv'
      print output1
      print 'mine'
      print output2
      print 'diff'
      print output1.shape, output2.shape
      print output1 - output2
      raise
  if len(scores1) == 0:
    assert len(scores2) == 0
  else:
    try:
      assert (scores1 == scores2).all()
    except:
      print 'opencv'
      print scores1
      print 'mine'
      print scores2
      print 'diff'
      print scores1.shape, scores2.shape
      print scores1 - scores2
      raise
  if t % (num_trials / 20) == 0:
    print 'passed', t
