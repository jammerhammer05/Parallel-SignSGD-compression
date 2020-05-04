#!/usr/bin/python3
#

import numpy as np

def compress(theta_grad):
	
	shape_theta = theta_grad.shape

	print("shape :", shape_theta)
	loops = shape_theta[0]
	itr = shape_theta[1]
	reslist = []
	
	while(loops > 0):

		res = 0
		itr = shape_theta[1]
		i = 0
		while(itr > 0):
	        
			x = int(theta_grad[loops-1][itr-1])
			if(x == -1):
				res =  res + 2 * pow(3,i)
			elif(x == 0):
				res =  res 
			elif(x == 1):
				res =  res + 1 * pow(3, i)
			else :
				print("Error in signSGD compression")
				exit(0)

			itr = itr - 1
			i = i + 1

		reslist.append(res)
		loops = loops - 1

	return shape_theta[1], reslist

def decompress(t_c, l):
    
	t_len = len(t_c)
	reslist = []
	i = 0

	while (i < t_len):

		res = []
		t = t_c[i]

		while t:
			t, r = divmod(t, 3)
			if(r == 2):
				res.append(-1)
			else:
				res.append(r)

		lenlist = len(res)

		if(lenlist != l):
			dif = l - lenlist
			while dif:
				res.append(0)
				dif = dif-1

		print(res)
		res.reverse()
		i = i+1
		reslist.append(res)

	reslist.reverse()
	resnp = np.array(reslist)
	return resnp

def compare(t1, t2):
	shape = t1.shape
	y = shape[0]
	
	while(y > 0):
		x = shape[1]
		
		while(x > 0):
			if(t1[y-1][x-1] != t2[y-1][x-1]):
				return False
			x = x-1
		y = y - 1
	
	return True

t1 = [[-1., -1., -1., -1., -1., 0., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
  -1., -1., -1., -1., -1., -1., -1., -1.],
 [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,
   1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]]

# t1 = [[0,1,-1],[1,0,-1],[0,0,1]]

t1_np = np.array(t1)
print(t1_np)

l, t1_comp = compress(t1_np)
print("t1_comp:\n", t1_comp)

t1_d = decompress(t1_comp, l)
print("the result is :\n",t1_d)
if(compare(t1_d, t1)):
	print("YES")
else :
	print("NO")
