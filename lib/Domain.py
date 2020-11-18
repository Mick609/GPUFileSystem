import uuid
from datetime import datetime
import pycuda.gpuarray as gpuarray
class pyGPUfsFile:
	def __init__(self, fileName, nbyte, fileType, absPath, gpuarray, size, lastAccessTime,lastModifyTime):
		self.fileName = fileName
		self.absPath = absPath
		self.nbyte = nbyte
		self.gpuarray = gpuarray
		self.size = size
		self.fileType = fileType

		#in nanosecond
		self.lastAccessTime = lastAccessTime
		self.lastModifyTime = lastModifyTime

		self.uuid = uuid.uuid4().hex

	def __str__(self):
		return  str('\t' + 'File Name:' + '\t\t' +			self.fileName + '\n' +
					'\t' + 'UUID:' + '\t\t\t' + 			self.uuid + '\n' +
					'\t' + 'Path:' + '\t\t\t' + 			self.absPath + '\n' +
					'\t' + 'Size:(Byte)' + '\t\t' + 		str(self.size) + '\n' +
					'\t' + 'Last Access Time:' + '\t' + 	str(datetime.fromtimestamp(self.lastAccessTime // 1000000000)) + '\n' +
					'\t' + 'Last Modify Time:' + '\t' + 	str(datetime.fromtimestamp(self.lastModifyTime // 1000000000)) + '\n')