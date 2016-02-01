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
        self.ecg_data = None
        self.samples_per_lead = None

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
                self.ecg_data = np.fromfile(
                    self.ecg_file, dtype=ecg_dtype, count=int(self.samples_per_lead))

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
            
    def plot_leads(self, leads=None, start_time=0, end_time=5):

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
        if self.ecg_data.size == 0:
            print("ECG data not loaded yet...")
            return

        # Loop through given leads to plot
        for index, lead in enumerate(leads):
            if lead != 1:
                pass
            else:
                print("Plotting lead: " + str(index))
                # Resolution[0] = 10000nv = 0.01mv
                y = self.ecg_data['samples'][:, index] * (self.resolution[index] / 1000000.0)
                x = np.linspace(0, self.samples_per_lead / self.sampling_rate,
                                num=self.samples_per_lead)
                plt.plot(x, y, label='lead ' + str(index))
                plt.title('ECG')
                plt.xlabel('s')
                plt.ylabel('mV')
                plt.xlim(start_time,end_time)
                plt.ylim(-1,1)
                plt.legend()
        plt.show()

        return

    def send_ecg_to_gpu(self, lead_number=None):
        """
        Compute a histogram on the ecg signal data
        """
        
        if not lead_number:
            print("Error in send_ecg_to_gpu(), please specify the \
                   lead number to compute a histogram on")
            return
            
        # Try to import the pycuda module
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            from pycuda.compiler import SourceModule
        except ImportError:
            print("Error: send_ecg_to_gpu() Unable to import the pyCUDA package")
            return
            
        module = SourceModule("""
        __global__ void histogram(float* slope_buffer, unsigned char * pix, int total)
        {
          int idx = threadIdx.x + blockDim.x * blockIdx.x;
          if(idx >= total) return;

          __shared__ float local_hist[256];

          if(threadIdx.x < 256)
            local_hist[threadIdx.x] = 0;

          __syncthreads();

          int pixel = pix[idx];
          atomicAdd((float *)&local_hist[pixel], 256.0/total);

          if(threadIdx.x < 256)
            atomicAdd((int*)&slope_buffer[threadIdx.x], local_hist[threadIdx.x]);
        }
        """)

        histogram = module.get_function("histograms")

    def compute_derivative_gpu(self, lead_number=None):
        """ Compute the derivatives of a given lead signal
        """
        # Try to import the pycuda module
        # try:
            # import pycuda.driver as cuda
            # import pycuda.autoinit
            # from pycuda.compiler import SourceModule
        # except ImportError:
            # print("Error: compute_derivative_gpu() Unable to import the pyCUDA package")
            # return
            
       
        
        y = self.ecg_data['samples'][:, lead_number] * (self.resolution[lead_number] / 1000000.0)
        x = np.linspace(0, self.samples_per_lead / self.sampling_rate,
                                num=self.samples_per_lead)
                                
                                
        dx = np.diff(x)
        #dxn = -np.diff(x)
        df = np.diff(y)/dx
        #threshold
        #plt.plot(x, y, label='lead ' + str(lead_number))
        
        #s=np.sort(df)
        #s0=s[0]*0.7
        #np.negative()
        #rr = np.zeros_like(s)
        #rr=[int(x>s0) for x in s ]
        plt.plot(x[:-1], df, label='df/dx ' + str(lead_number))
        #plt.plot(x[:-1], s, label='df/dx ' + str(lead_number))
        #plt.plot(x[:-1],rr, label='rr', marker='o')
       
        plt.title('ECG')
        plt.xlabel('s')
        plt.ylabel('mV')
        plt.xlim(0,5)
        #plt.ylim(-1,1)
        plt.legend()
        
        
        plt.show()
       # for i, s in np.ndenumerate(df):
        #    print(s)


        
        return