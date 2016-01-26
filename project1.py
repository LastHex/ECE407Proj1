import sys

# Custom imports
import patient

filename = sys.argv[1]
try:
    f = open(filename, 'rb')
except IOError:
    print('%s cannot be opened', filename)
    sys.exit()

# Create the patient object
p = patient.Patient()

# Read the file
p.load_ecg_data(filename)

# Plot the file
p.plot_leads()