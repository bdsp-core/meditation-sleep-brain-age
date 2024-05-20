# List1
#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
# import numpy as np
import glob
import datetime
# import mne
# importing required modules
import PyPDF2

# creating a pdf file object
pdfFileObj = open('example.pdf', 'rb')

# creating a pdf reader object
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

# printing number of pages in pdf file
print(pdfReader.numPages)

# creating a page object
pageObj = pdfReader.getPage(0)

# extracting text from page
print(pageObj.extractText())

# closing the pdf file object
pdfFileObj.close()

# # rename txt and png
files = glob.glob(os.getcwd() + '\\' + 'MEDITATION_med_data' + '/*.txt')
date_dic = {}
dic = {}
grab_lines = False
delete_list = ['Scorer Time:']
for file in files:
     with open(file, 'r') as sleep_data_meditation_file:
        for line in sleep_data_meditation_file:
            if line.startswith('Scorer Time:'):
                grab_lines = True
                for word in delete_list:
                    print(word)
                    line = line.replace(word, "")
                    print("Line:", line)
                    line = line.strip()
                    new_name = line.split(':')[0] + line.split(':')[1] + line.split(':')[2]
                    new_name = new_name.split('/')[0] + new_name.split('/')[1] + new_name.split('/')[2]
                    print("New Name:", new_name)
        sleep_data_meditation_file.close()
        os.rename(file, os.getcwd() + '\\' + 'MEDITATION_med_data' + '\\' + new_name + '.txt')



