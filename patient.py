from __future__ import division, print_function  # python 2/3 interop
import numpy as np
import matplotlib.pyplot as plt


class Patient:
    """
    Class that handles the patient info and ecg files
    """

    def __init__(self):
        self.magic_number = None
        self.checksum = None
        self.var_length_block_size = None
        self.sample_size_ecg = None
        self.offset_var_length_block = None
        self.offset_ecg_block = None
        self.file_version = None
        self.first_name = None
        self.last_name = None
        self.ID = None
        self.sex = None
        self.race = None
        self.birth_date = None
        self.record_date = None
        self.file_date = None
        self.start_time = None
        self.n_leads = None
        self.lead_spec = None
        self.lead_quality = None
        self.resolution = None
        self.pacemaker = None
        self.recorder = None
        self.sampling_rate = None
        self.proprietary = None
        self.copyright = None
        self.reserved = None
        self.dt = None
        self.var_block = None
        self.ecg_file = None
        self.samples_per_lead = None
        self.active_leads = None
        self.ecg_lead_derivatives = {}
        self.ecg_lead_voltages = {}
        self.ecg_time_data = None
        self.ecg_data_loaded = False
        self.heart_rate = {}
        self.lead_rr = {}
        return

    def load_ecg_header(self, filename):
        """
        Open the ECG file and only read the header
        :param filename: Name of the ECG file
        :return:
        """

        try:
            with open(filename, 'rb') as self.ecg_file:
                print("Reading filename (header only): " + filename)

                self._get_header_data()

        except IOError:
            print("File cannot be opened:", filename)

    def load_ecg_data(self, filename):
        """
        Open the ECG file and read the data
        :param filename: path name of the file to read
        """

        try:
            with open(filename, 'rb')as self.ecg_file:
                print("Reading filename (header and data): " + filename)

                self._get_header_data()

                # Set the datatype to load all of the samples in one chunk
                ecg_dtype = np.dtype([('samples', np.int16, self.n_leads)])
                ecg_data = np.fromfile(
                    self.ecg_file, dtype=ecg_dtype, count=int(self.samples_per_lead))

                # Put the ecg data into a dictionary
                for index, lead in np.ndenumerate(self.lead_spec):
                    if lead == 1:
                        self.ecg_lead_voltages[index[0]] = ecg_data['samples'][:, index[0]]
                    else:
                        self.ecg_lead_voltages[index[0]] = None

                self.ecg_data_loaded = True
                self.active_leads = [i for i, x in enumerate(self.lead_spec) if x == 1]
                self.ecg_time_data = np.linspace(0, self.samples_per_lead / self.sampling_rate,
                                                 num=self.samples_per_lead)

                a = np.datetime64(str(self.record_date[2]) + '-' +
                  '{0:02d}'.format(self.record_date[1]) + '-'+
                  '{0:02d}'.format(self.record_date[0]) + 'T' +
                  '{0:02d}'.format(self.start_time[0]) + ':' +
                  str(self.start_time[1]) + ':00.000')

                self.dt_datetime = np.arange(start=a,
                               stop=a + np.timedelta64(int(self.samples_per_lead) * 5, 'ms'),
                               step=np.timedelta64(5, 'ms'))

        except IOError:
            print("File cannot be opened:", filename)

    def _get_header_data(self):
        self.magic_number = np.fromfile(
            self.ecg_file, dtype=np.dtype('a8'), count=1)[0]
        self.checksum = np.fromfile(self.ecg_file, dtype=np.uint16, count=1)[0]
        self.var_length_block_size = np.fromfile(
            self.ecg_file, dtype=np.int32, count=1)[0]
        self.sample_size_ecg = np.fromfile(
            self.ecg_file, dtype=np.int32, count=1)[0]
        self.offset_var_length_block = np.fromfile(
            self.ecg_file, dtype=np.int32, count=1)[0]
        self.offset_ecg_block = np.fromfile(
            self.ecg_file, dtype=np.int32, count=1)[0]
        self.file_version = np.fromfile(
            self.ecg_file, dtype=np.int16, count=1)[0]
        self.first_name = np.fromfile(
            self.ecg_file, dtype=np.dtype('a40'), count=1)[0]
        self.last_name = np.fromfile(
            self.ecg_file, dtype=np.dtype('a40'), count=1)[0]
        self.ID = np.fromfile(self.ecg_file, dtype=np.dtype('a20'), count=1)[0]
        self.sex = np.fromfile(self.ecg_file, dtype=np.int16, count=1)[0]
        self.race = np.fromfile(self.ecg_file, dtype=np.int16, count=1)[0]
        self.birth_date = np.fromfile(self.ecg_file, dtype=np.int16, count=3)
        self.record_date = np.fromfile(self.ecg_file, dtype=np.int16, count=3)
        self.file_date = np.fromfile(self.ecg_file, dtype=np.int16, count=3)
        self.start_time = np.fromfile(self.ecg_file, dtype=np.int16, count=3)
        self.n_leads = np.fromfile(self.ecg_file, dtype=np.int16, count=1)[0]
        self.lead_spec = np.fromfile(self.ecg_file, dtype=np.int16, count=12)
        self.lead_quality = np.fromfile(self.ecg_file, dtype=np.int16, count=12)
        self.resolution = np.fromfile(self.ecg_file, dtype=np.int16, count=12)
        self.pacemaker = np.fromfile(self.ecg_file, dtype=np.int16, count=1)[0]
        self.recorder = np.fromfile(
            self.ecg_file, dtype=np.dtype('a40'), count=1)[0]
        self.sampling_rate = np.fromfile(
            self.ecg_file, dtype=np.int16, count=1)[0]
        self.proprietary = np.fromfile(
            self.ecg_file, dtype=np.dtype('a80'), count=1)[0]
        self.copyright = np.fromfile(
            self.ecg_file, dtype=np.dtype('a80'), count=1)[0]
        self.reserved = np.fromfile(
            self.ecg_file, dtype=np.dtype('a88'), count=1)[0]
        if self.var_length_block_size > 0:
            self.dt = np.dtype((str, self.var_length_block_size))
            self.var_block = np.fromfile(self.ecg_file, dtype=self.dt, count=1)[0]
        self.samples_per_lead = self.sample_size_ecg / self.n_leads

    def plot_ecg_lead_voltage(self, leads=None, start_time=0, end_time=5):

        """
        Plot the ecg leads (0 ordered!). Defaults to plot all of the leads
        :param leads: nothing (default to 1), integer of lead number, or list of ints
        :param start_time: starting time to plot in seconds (default 0)
        :param end_time: ending time to plot (default 5)
        """
        if leads is None:
            print("No lead number or list provided, defaulting to plot all...")
            leads = self.lead_spec
        elif isinstance(leads, list):
            _lead = leads
            leads = np.zeros_like(self.lead_spec)
            for l in _lead:
                leads[l] = 1
        elif isinstance(leads, int):
            _lead = leads
            leads = np.zeros_like(self.lead_spec)
            leads[_lead] = 1
        else:
            print('Error: plot_leads() argument must be int, a list of ints, or empty for all leads')
            return

        # Try to import matplotlib
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed")
            return

        # Check to make sure ecg data is loaded
        if not self.ecg_data_loaded:
            print("ECG data not loaded yet...")
            return

        # Loop through given leads to plot
        for index, lead in enumerate(leads):
            if lead != 1:
                pass
            else:
                print("Plotting lead: " + str(index))
                # Resolution[0] = 10000nv = 0.01mv
                y = self.ecg_lead_voltages[index] * (self.resolution[index] / 1000000.0)
                x = np.linspace(0, self.samples_per_lead / self.sampling_rate,
                                num=self.samples_per_lead)
                plt.plot(x, y, label='lead ' + str(index))
                plt.title('ECG')
                plt.xlabel('s')
                plt.ylabel('mV')
                plt.xlim(start_time, end_time)
                plt.ylim(-1, 1)
                plt.legend()
        plt.show()

        return

    def plot_ecg_lead_derivative(self, leads=None, start_time=0, end_time=5):
        """
        Plot the ecg leads (0 ordered!). Defaults to plot all of the leads
        :param leads: nothing (default to 1), integer of lead number, or list of ints
        :param start_time: starting time to plot in seconds (default 0)
        :param end_time: ending time to plot (default 5)
        """
        if leads is None:
            print("No lead number or list provided, defaulting to plot all...")
            leads = self.lead_spec
        elif isinstance(leads, list):
            _lead = leads
            leads = np.zeros_like(self.lead_spec)
            for l in _lead:
                leads[l] = 1
        elif isinstance(leads, int):
            _lead = leads
            leads = np.zeros_like(self.lead_spec)
            leads[_lead] = 1
        else:
            print('Error: plot_leads() argument must be int, a list of ints, or empty for all leads')
            return

        # Try to import matplotlib
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed")
            return

        # Check to make sure ecg data is loaded
        if not self.ecg_data_loaded:
            print("ECG data not loaded yet...")
            return

        # Loop through given leads to plot
        for index, lead in enumerate(leads):
            if lead != 1:
                pass
            else:
                print("Plotting lead: " + str(index))
                # Resolution[0] = 10000nv = 0.01mv
                y = self.ecg_lead_derivatives[index][:5000] * (self.resolution[index] / 1000000.0)

                plt.plot(x=y, y=self.ecg_time_data[1:5001], label='lead ' + str(index))
                plt.title('ECG')
                plt.xlabel('s')
                plt.ylabel('mV')
                plt.xlim(start_time, end_time)
                plt.ylim(-1, 1)
                plt.legend()
        plt.show()

        return

    def plot_leads_s(self, leads=None, start_time=0, end_time=5):
        """
        Plot the ecg leads (0 ordered!). Defaults to plot all of the leads
        :param leads: nothing (default to 1), integer of lead number, or list of ints
        :param start_time: starting time to plot in seconds (default 0)
        :param end_time: ending time to plot (default 5)
        """

        if leads is None:
            print("No lead number or list provided, defaulting to plot all...")
            leads = self.lead_spec
        elif isinstance(leads, list):
            _lead = leads
            leads = np.zeros_like(self.lead_spec)
            for l in _lead:
                leads[l] = 1
        elif isinstance(leads, int):
            _lead = leads
            leads = np.zeros_like(self.lead_spec)
            leads[_lead] = 1
        else:
            print('Error: plot_leads() argument must be int, a list of ints, or empty for all leads')
            return

        # Try to import matplotlib
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed")
            return

        # Check to make sure ecg data is loaded
        if not self.ecg_data_loaded:
            print("ECG data not loaded yet...")
            return

        # Loop through given leads to plot
        for index, lead in enumerate(leads):
            if lead != 1:
                pass
            else:
                print("Plotting lead: " + str(index))
                # Resolution[0] = 10000nv = 0.01mv
                y = self.ecg_lead_voltages[index] * (self.resolution[index] / 1000000.0)
                x = np.linspace(0, self.samples_per_lead / self.sampling_rate,
                                num=self.samples_per_lead)
                y.sort()
                plt.plot(x, y, label='lead ' + str(index))
                plt.title('ECG')
                plt.xlabel('s')
                plt.ylabel('mV')
                plt.xlim(0, 3600 * 24)
                plt.ylim(-1, 1)
                plt.legend()
        plt.show()

        return

    def compute_lead_derivative_cpu(self, lead_number=None):
        """ 
        Compute the derivatives of a given lead signal
        :param lead_number: Intger of the lead number to calculate the derivative of
        :return: Returns an array of the derivatives from the specified lead number
        """

        # Check to make sure the given lead number is valid
        if self.lead_spec[lead_number] == 1:
            pass
        else:
            print("Error: compute_derivative_gpu(); Given lead number", lead_number, "is not valid ")
            print("Valid lead numbers: ", [i for i, x in enumerate(self.lead_spec) if x == 1])
            return

        print("Computing derivative for lead", lead_number, "on the cpu...")
        f = self.ecg_lead_voltages[lead_number] * (self.resolution[lead_number] / 1000000.0)
        # x = np.linspace(0, self.samples_per_lead / self.sampling_rate,
        #                num=self.samples_per_lead)
        dx = self.samples_per_lead / self.sampling_rate / self.samples_per_lead

        # dx = np.diff(x)
        self.ecg_lead_derivatives[lead_number] = np.diff(f) / dx

    def compute_lead_derivative_gpu(self, lead_number=None):
        """ 
        Compute the derivatives of a given lead signal
        :param lead_number: Integer of the lead number to calculate the derivative of
        :return: Returns an array of the derivatives from the specified lead number
        """

        # Try to import the pyCUDA module
        try:
            from pycuda import driver, compiler, gpuarray, tools, cumath
            import pycuda.autoinit
            from pycuda.compiler import SourceModule
        except ImportError:
            print("Error: compute_derivative_gpu(); Unable to import the pyCUDA package")
            return

        # Check to make sure the given lead number is valid
        if self.lead_spec[lead_number] == 1:
            pass
        else:
            print("Error: compute_derivative_gpu(); Given lead number", lead_number, "is not valid ")
            print("Valid lead numbers: ", [i for i, x in enumerate(self.lead_spec) if x == 1])
            return

        print("Computing derivative for lead", lead_number, "on the gpu...")

        lead_voltage = self.ecg_lead_voltages[lead_number] * (self.resolution[lead_number] / 1000000.0)
        lead_voltage = lead_voltage.astype(np.float32)
        dx = np.float32(self.samples_per_lead / self.sampling_rate / self.samples_per_lead)

        # Create the gpu arrays
        lead_voltage_gpu = gpuarray.to_gpu(lead_voltage[:-1])
        lead_voltage_gpu_shifted = gpuarray.to_gpu(lead_voltage[1:])

        dfdx_gpu = gpuarray.zeros(lead_voltage[1:].size, np.float32)
        n = lead_voltage[1:].size

        # Device info
        gpu_dev = tools.DeviceData()
        threads_per_block = 512  # gpu_dev.max_threads

        # CUDA kernels

        #with open('kernels.cu','r') as f:
        #    module = SourceModule(f.read())

        module = SourceModule("""
        #include <stdio.h>
        // Simple gpu kernel to compute the derivative with a constant dx
        __global__ void derivative_shared(float* f, float* f_shifted, float* dfdx, float dx, int n)
        {
            unsigned int g_i = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int s_i = threadIdx.x;

            __shared__ float shared_dx;
            __shared__ float shared_f[512];
            __shared__ float shared_f_shifted[512];

            // Assigned to shared memory
            shared_dx = dx;
            shared_f[s_i] = f[g_i];
            shared_f_shifted[s_i] = f_shifted[g_i];

            __syncthreads();

            dfdx[g_i] = (shared_f_shifted[s_i] - shared_f[s_i]) / shared_dx;
        }

        __global__ void derivative(float* f, float* f_shifted, float* dfdx, float dx, int n)
        {
            unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            dfdx[i] = (f_shifted[i] - f[i]) / dx;
        }

        // Filter out values that are not big enough (use for very large derivatives) i.e. large positive spikes
        __global__ void get_only_large_deriv(float* f, int* thresholded_f, float threshold)
        {
            __shared__ float shared_f[512];
            __shared__ int   shared_thresholded_f[512];
            __shared__ float shared_threshold;

            unsigned int g_i = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned int b_i = threadIdx.x;

            // Put the threshold value into shared memory
            shared_threshold = threshold;

            // Put f into shared memory
            shared_f[b_i] = f[g_i];

            __syncthreads();

            // 1 if at or above threshold, 0 if not
            shared_thresholded_f[b_i] = (shared_f[b_i] >= shared_threshold) ? 1 : 0;

            // Put threshold values into global memory
            thresholded_f[g_i] = shared_thresholded_f[b_i];

        }
        """)

        n_blocks = np.int_(np.ceil(n / threads_per_block))

        # print('Launching', n_blocks, 'blocks with', threads_per_block, 'threads per block for a total of',
        #      n_blocks * threads_per_block, 'threads')


        # Calculate the first derivative
        derivative_kernel = module.get_function("derivative_shared")

        print("Calling derivative kernel...")
        derivative_kernel(lead_voltage_gpu,
                          lead_voltage_gpu_shifted,
                          dfdx_gpu,
                          dx,
                          np.int_(n),
                          block=(threads_per_block, 1, 1),
                          grid=(n_blocks, 1, 1))

        print("Done")
        thresh = 0.7
        max_dfdx = gpuarray.max(dfdx_gpu)
        # min_dfdx = gpuarray.min(dfdx_gpu)

        print("Max dfdx", max_dfdx)

        threshold_value = thresh * max_dfdx
        print("Using", threshold_value, "to threshold")

        # binary array (1 or 0) 1 means r spike, 0 means none
        rr_binary_array = gpuarray.empty(lead_voltage[1:].size, np.int)

        # Get the large positive derivative spikes
        threshold_kernel = module.get_function("get_only_large_deriv")

        #n_blocks = np.int_(np.ceil(lead_voltage[1:].size / threads_per_block))

        print("Calling threshold kernel...")
        threshold_kernel(dfdx_gpu,
                         rr_binary_array,
                         threshold_value,
                         block=(threads_per_block, 1, 1),
                         grid=(n_blocks, 1, 1))

        # print('cumath.fabs', gpuarray.max(cumath.fabs(dfdx_gpu)))
        print("Done")

        self.lead_rr[lead_number] = rr_binary_array.get()
        np.savez('rr_lead' + str(lead_number) + '.npz', lead=lead_number, rr=self.lead_rr[lead_number])
        # return rr_binary_array.get()
        # return dfdx_gpu.get()

    def save_lead_ecg_pickle(self, lead_number=None, filename=None):
        """
        Save the ecg data as a numpy pickle file
        :param filename: Name of the pickle file (Defaults to lead_N.npy)
        :param lead_number: ecg lead number
        :return:
        """

        if filename is None:
            filename = 'ecg_lead_' + str(lead_number)
        if lead_number is None:
            print("Error in save_lead_ecg_pickle: Please specify a lead number (int)")
            print("Valid lead numbers: ", [i for i, x in enumerate(self.lead_spec) if x == 1])
            return

        # Check to make sure the given lead number is valid
        if self.lead_spec[lead_number] == 1:
            pass
        else:
            print("Error: compute_derivative_gpu(); Given lead number", lead_number, "is not valid ")
            print("Valid lead numbers: ", [i for i, x in enumerate(self.lead_spec) if x == 1])
            return

        print("Attempting to save the lead ecg data to file...")
        try:
            np.savez(filename, lead=lead_number, ecg_data=self.ecg_lead_voltages[lead_number])
            print('Saved to ', filename)
        except KeyError:
            print('Error: save_lead_ecg_pickle(); Derivative doesnt exist for this lead number')
            return

    def load_lead_ecg_pickle(self, filename=None):
        """
        Load the lead ecg data numpy pickle file
        :param filename: name of the file to open
        """

        if filename is None:
            print("Error: load_lead_ecg_pickle; No file name given..")
            return

        with np.load(filename) as f:
            lead_number = f['lead']
            print('Loading ecg data for lead', lead_number, 'into memory...')
            try:
                self.ecg_lead_voltages[int(lead_number)] = f['ecg_data']
                print('Done')
            except KeyError:
                print('Failed... the format is most likely wrong...')

    def save_lead_derivative_pickle(self, lead_number=None, filename=None):
        """
        Save the derivative array to a numpy pickle file
        :param filename: Name of the pickle file (Defaults to lead_N_derivative.npy)
        :param lead_number: ecg lead number
        :return:
        """

        if filename is None:
            filename = 'ecg_lead_deriv' + str(lead_number)

        if lead_number is None:
            print("Error: save_lead_derivative_pickle(); Please specify a lead number (int)")
            print("Valid lead numbers: ", [i for i, x in enumerate(self.lead_spec) if x == 1])
            return

        # Check to make sure the given lead number is valid
        if self.lead_spec[lead_number] == 1:
            pass
        else:
            print("Error: save_lead_derivative_pickle(); Given lead number", lead_number, "is not valid ")
            print("Valid lead numbers: ", [i for i, x in enumerate(self.lead_spec) if x == 1])
            return

        print("Attempting to save the lead derivative data to file...")
        try:

            np.savez(filename, lead=lead_number, deriv_data=self.ecg_lead_derivatives[lead_number])
            print('Saved to ', filename)
        except KeyError:
            print('Error: save_lead_derivative_pickle(); Derivative doesnt exist for this lead number')
            return

    def load_lead_derivative_pickle(self, filename=None):
        """
        Load the derivative numpy pickle file
        :param filename: name of the file to open
        """

        if filename is None:
            print("Error: load_lead_derivative_pickle; No file name given..")
            return

        with np.load(filename) as f:
            lead_number = f['lead']
            print('Loading derivatives for lead', lead_number, 'into memory...')
            try:
                self.ecg_lead_derivatives[int(lead_number)] = f['deriv_data']
                print('Done')
            except KeyError:
                print('Failed... the format is most likely wrong...')

    def find_rr_peaks(self, lead_number=None):
        """ Compute the derivatives of a given lead signal
        """

        print("Finding rr peaks for lead number", lead_number, "...")

        df = self.ecg_lead_derivatives[lead_number]

        dfm1 = 0
        large = 0
        loc = 0
        # if(abs(max(df))>abs(min(df))):
        #	thresh = max(df)*.7
        # else:
        #	thresh = min(df)*.7
        peaks = []
        peaks_location = []
        last = 0
        twoago = 0
        k = 0
        for i in df:
            if ((k % 100000 == 0) & (len(df) - (100000 + k) >= 0)):
                for x in range(0, 100000):
                    if (abs(df[k + x]) > abs(large)):
                        large = df[k + x]
                        thresh = df[k + x] * .7
                        # print("Peaks with threshold " + repr(thresh) + ":\n")
            if (abs(last) >= abs(thresh)):
                if (abs(last) > abs(twoago)):
                    if (abs(last) > abs(i)):
                        peaks.append(last)
                        peaks_location.append(loc)
                        # print(repr(last) + " : " + repr(loc) + "\n")
            twoago = last
            last = i
            k += 1
            large = 0;
            loc += (self.samples_per_lead / self.sampling_rate) / self.samples_per_lead
        RR = 0
        HR = []
        last = 0
        k = 0

        for i in peaks_location:
            if (k > 0):
                if (((60 / (i - last)) > 200) | ((60 / (i - last)) < 30)):
                    HR.append(np.nan)
                else:
                    HR.append(60 / (i - last))
            # print(repr (i) + "-" + repr(last) + "=" + repr((i-last)) + "\n")
            k += 1
            last = i
        xh = np.linspace(0, 3600 * 24, num=len(HR))

        self.heart_rate[lead_number] = [None, None]
        self.heart_rate[lead_number][1] = np.copy(np.array(HR))
        self.heart_rate[lead_number][0] = np.copy(xh)


