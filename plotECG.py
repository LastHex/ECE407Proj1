import sys
import numpy as np
import struct
import matplotlib.pyplot as plt




if __name__ == "__main__":
	
	filename = sys.argv[1]
	
	
	try: 
		f = open(filename,'rb')
		
	except IOError:
		print '%s cannot be opened' % filename
		sys.exit()
		
	else:

		# magic number
		magicnumber = np.fromfile(f, dtype = np.dtype('a8'), count = 1)[0]
		
		# check sum
		chesksum = np.fromfile(f, dtype = np.uint16, count = 1)[0]
	
		#header
		Var_length_block_size = np.fromfile(f, dtype = np.int32, count = 1)[0] 
		Sample_Size_ECG =  np.fromfile(f, dtype = np.int32, count = 1)[0] 
		Offset_var_length_block =  np.fromfile(f, dtype = np.int32, count = 1)[0] 
		Offset_ECG_block =  np.fromfile(f, dtype = np.int32, count = 1)[0] 
		File_version =  np.fromfile(f, dtype = np.int16, count = 1)[0] 
		First_name =  np.fromfile(f, dtype = np.dtype('a40'), count = 1)[0]
		Last_name =  np.fromfile(f, dtype = np.dtype('a40'), count = 1)[0]
		ID = np.fromfile(f, dtype = np.dtype('a20'), count = 1)[0]
		Sex = np.fromfile(f, dtype = np.int16, count = 1)[0] 
		Race = np.fromfile(f, dtype = np.int16, count = 1)[0] 
		Birth_Date = np.fromfile(f, dtype = np.int16, count = 3) 
		Record_Date =  np.fromfile(f, dtype = np.int16, count = 3) 
		File_Date =  np.fromfile(f, dtype = np.int16, count = 3) 
		Start_Time =  np.fromfile(f, dtype = np.int16, count = 3) 
		nbLeads = np.fromfile(f, dtype = np.int16, count = 1)[0] 
		Lead_Spec = np.fromfile(f, dtype = np.int16, count = 12) 
		Lead_Qual = np.fromfile(f, dtype = np.int16, count = 12) 
		Resolution = np.fromfile(f, dtype = np.int16, count = 12) 
		Pacemaker = np.fromfile(f, dtype = np.int16, count = 1)[0] 
		Recorder =  np.fromfile(f, dtype = np.dtype('a40'), count = 1)[0]
		Sampling_Rate = np.fromfile(f, dtype = np.int16, count = 1)[0] 
		Propreitary  = np.fromfile(f, dtype = np.dtype('a80'), count = 1)[0]
		Copyright =  np.fromfile(f, dtype = np.dtype('a80'), count = 1)[0]
		Reserved =  np.fromfile(f, dtype = np.dtype('a88'), count = 1)[0]
	
		# read Variable length block
		if (Var_length_block_size >0):
			dt = dtype((str,Var_length_block_size))
			varblock = np.fromfile(f, dtype = dt, count = 1)[0]
		
		# ECG data
		Sample_per_lead = Sample_Size_ECG/nbLeads
		
		ecgSig = np.zeros((nbLeads, Sample_per_lead))
		
		for i in range(Sample_per_lead):
			for j in range(nbLeads):
				ecgSig[j][i] =  np.fromfile(f, dtype = np.int16, count = 1)[0] 
		
		# print parameters
		print 'Sample Size = %d' % Sample_Size_ECG
		print 'Number of Leads = %d' % nbLeads
		print 'Resolution = %s' % Resolution
		print 'Sampling Rate = %d' % Sampling_Rate
		
		f.close()
		
		
# plotting ECG

		# Resolution[0] = 10000nv = 0.01mv		
		y1 = ecgSig[0][:] * (Resolution[0]/1000000.0)
		y2 = ecgSig[1][:] * (Resolution[1]/1000000.0)
		y3 = ecgSig[2][:] * (Resolution[2]/1000000.0)
		
		x = np.linspace(0, Sample_per_lead/Sampling_Rate ,num = Sample_per_lead)
		# plot of three leads
	 	#plt.figure(1)
	 	#plt.plot(x,y1, 'r',x,y2, 'b',x,y3,  'g')
		#plt.title('ECG')
		#plt.xlabel('s')
		#plt.ylabel('mV')
		#plt.xlim(0,5)
		#plt.ylim(-1.5,1)
		
		# plot of one lead
		plt.figure(2)
		z = ecgSig[1][:] * (Resolution[0]/1000000.0)
		plt.plot(x,z, color='b',marker='o')
		plt.title('ECG')
		plt.xlabel('s')
		plt.ylabel('mV')
		plt.xlim(0,5)
		plt.ylim(-1,1)
		
		
	

		
		
		plt.show()
	

	