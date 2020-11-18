import os.path
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import numpy as np
from lib import Domain
import base64
from PIL import Image

class Global:
	fileList = []
	lastShownList = []

def editDistance(str1, str2,m,n): 
    if m == 0: 
        return n 
  
    if n == 0: 
        return m 
  
    if str1[m-1]== str2[n-1]: 
        return editDistance(str1, str2, m-1, n-1) 
  
    return 1 + min(editDistance(str1, str2, m, n-1),    # Insert 
                   editDistance(str1, str2, m-1, n),    # Remove 
                   editDistance(str1, str2, m-1, n-1)    # Replace 
                   ) 

def store(filePath):
	#store a file to the GPU memory
	if os.path.isfile(filePath):
		im = Image.open(filePath)
		im = im.convert(mode='RGB', colors=256)
		'''
		a = np.array(im)
		a_gpu = cuda.mem_alloc(a.nbytes)
		cuda.memcpy_htod(a_gpu, a)
		'''
		a = np.array(im)
		a = a.astype('float64')
		gpuarrayInst = gpuarray.to_gpu(a)

		fileName = os.path.basename(filePath)
		nbyte = a.nbytes
		absPath = os.path.abspath(filePath)
		size = (a.shape)
		lastAccessTime = os.stat(absPath).st_atime_ns
		lastModifyTime = os.stat(absPath).st_mtime_ns
		Global.fileList.append(Domain.pyGPUfsFile(
			fileName = fileName, 
			absPath = absPath, 
			nbyte = nbyte, 
			fileType = 'image', 
			gpuarray = gpuarrayInst, 
			size = size, 
			lastAccessTime = lastAccessTime,
			lastModifyTime = lastModifyTime))
	else:
		raise RuntimeError('Cannot find file with path: ' + filePath)

def list():
	print('list()')
	Global.lastShownList = []
	for i in range(len(Global.fileList)):
		Global.lastShownList.append(Global.fileList[i])
	printList(Global.lastShownList)
	return Global.lastShownList

def printList(list):
	print('=============================')
	for i in range(len(Global.lastShownList)):
		print(i, Global.lastShownList[i])
	print('================================================================================')
	print()

def findByType(fileType):
	print('findByType('+fileType+')')
	Global.lastShownList = []
	for i in range(len(Global.fileList)):
		if Global.fileList[i].fileType == fileType:
			Global.lastShownList.append(Global.fileList[i])
	printList(Global.lastShownList)
	return Global.lastShownList

def findByName(fileName):
	print('findByName('+fileName+')')
	Global.lastShownList = []
	distances = []
	for i in range(len(Global.fileList)):
		distances.append(editDistance(fileName, Global.fileList[i].fileName, len(fileName), len(Global.fileList[i].fileName)))
	for i in range(10):
		minimal = -1
		if i < len(distances):
			Global.lastShownList.append(Global.fileList[distances.index(min(distances))])
			distances[distances.index(min(distances))] = float("inf")
	printList(Global.lastShownList)
	return Global.lastShownList

def readByIndex(index, destination):
	print('readByIndex('+str(index)+','+destination+')')
	if index < len(Global.lastShownList):
		opt = Global.lastShownList[index].gpuarray.get()
		opt = opt.astype('uint8')
		new_im = Image.fromarray(opt)
		new_im.save(destination)
	else:
		raise RuntimeError('Index out of range: ' + str(index) + ' with size: ' + str(len(Global.lastShownList)))

def writeByIndex(index, src):
	print('writeByIndex('+str(index)+','+src+')')
	if index < len(Global.lastShownList):
		Global.lastShownList[index].gpuarray.gpudata.free()

		if os.path.isfile(src):
			im = Image.open(src)
			im = im.convert(mode='RGB', colors=256)
			a = np.array(im)
			a = a.astype('float64')
			gpuarrayInst = gpuarray.to_gpu(a)

			fileName = os.path.basename(src)
			nbyte = a.nbytes
			fileType = 'image'
			absPath = os.path.abspath(src)
			#get file size in byte
			size = a.shape
			lastAccessTime = os.stat(absPath).st_atime_ns
			lastModifyTime = os.stat(absPath).st_mtime_ns

			#update info
			Global.lastShownList[index].fileName = fileName
			Global.lastShownList[index].absPath = absPath
			Global.lastShownList[index].nbyte = nbyte
			Global.lastShownList[index].fileType = fileType
			Global.lastShownList[index].gpuarray = gpuarrayInst
			Global.lastShownList[index].size = size
			Global.lastShownList[index].lastAccessTime = lastAccessTime
			Global.lastShownList[index].lastModifyTime = lastModifyTime
		else:
			raise RuntimeError('Cannot find file with path: ' + src)
	else:
		raise RuntimeError('Index out of range: ' + str(index) + ' with size: ' + str(len(Global.lastShownList)))

def freeFileByIndex(index):
	print('freeFileByIndex('+str(index)+')')
	if index < len(Global.lastShownList):
		Global.lastShownList[index].gpuarray.gpudata.free()
		Global.fileList.remove(Global.lastShownList[index])
	else:
		raise RuntimeError('Index out of range: ' + str(index) + ' with size: ' + str(len(Global.lastShownList)))

def compare(file_1, file_2):
	if file_1.gpuarray.shape == file_2.gpuarray.shape:
		num_values = file_1.gpuarray.shape[0] * file_1.gpuarray.shape[1] * file_1.gpuarray.shape[2]
		gpuarrayA = file_1.gpuarray
		gpuarrayB = file_2.gpuarray
		mean_diff = gpuarray.sum(abs(gpuarrayA - gpuarrayB)) / (num_values)
		return mean_diff.get()
	else:
		raise RuntimeError('Files in comparison have to have the same shape.')
