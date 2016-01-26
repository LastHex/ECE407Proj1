import numpy as np


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

                # read variable length block
                if self.var_length_block_size > 0:
                    self.dt = np.dtype((str, self.var_length_block_size))
                    self.var_block = np.fromfile(self.ecg_file, dtype=self.dt, count=1)[0]

                self.samples_per_lead = self.sample_size_ecg / self.n_leads

        except IOError:
            print("File cannot be opened:", filename)

    def load_ecg_data(self, filename):
        """
        Open the ECG file and read the data
        :param self:
        :param filename:
        :return:
        """

        if not self.ecg_file:
            # Open and read the header
            self.load_ecg_header(filename)
        else:
            try:
                self.ecg_file = open(filename, 'rb')
                print("Reading filename (header and data): " + filename)

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

                # read variable length block
                if self.var_length_block_size > 0:
                    self.dt = np.dtype((str, self.var_length_block_size))
                    self.var_block = np.fromfile(self.ecg_file, dtype=self.dt, count=1)[0]

                # ECG data
                self.samples_per_lead = self.sample_size_ecg / self.n_leads
                self.ecg_data = np.zeros((self.n_leads, self.samples_per_lead))

                # Loop through data (not efficient at all...)
                # TODO: Fix this to read all in one chunk and split once in memory
                for i in range(int(self.samples_per_lead)):
                    for j in range(self.n_leads):
                        self.ecg_data[j][i] = np.fromfile(self.ecg_file, dtype=np.int16, count=1)[0]

            except IOError:
                print("File cannot be opened:", filename)

    def plot_leads(self, leads):
        """
        Plot the ecg leads (0 ordered!). Defaults to lead 1 if none given
        :param self:
        :param leads: nothing (default to 1), integer of lead number, or list of ints
        """

        if leads is None:
            print("No lead number or list provided, defaulting to lead 1...")
            leads = 1

        if isinstance(leads, list):
            pass

        elif isinstance(leads, int):
            # Make it a list of a single for easier logic below
            _lead = leads
            leads = [_lead]

        # Try to import matplotlib
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed")
            return

        # Check to make sure ecg data is loaded
        if not self.ecg_data.any():
            print("ECG data not loaded yet...")
            return

        # Loop through given leads to plot
        for lead in leads:
            # Resolution[0] = 10000nv = 0.01mv
            y = self.ecg_data[lead - 1][:] * (self.resolution[lead - 1] / 1000000.0)
            x = np.linspace(0, self.samples_per_lead / self.sampling_rate,
                            num=self.samples_per_lead)
            plt.plot(x,y)
            plt.title('ECG')
            plt.xlabel('s')
            plt.ylabel('mV')
            plt.xlim(0, 5)
            plt.ylim(-1, 1)

        plt.show()

        return
