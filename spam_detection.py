# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 23:04:23 2023

@author: soham
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu


# loading the saved models

spam_model = pickle.load(open('spam_email.sav', 'rb'))
vectorizer_model = pickle.load(open('vectorizer.sav','rb'))

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Multiple ML and Deep Learning Projects',
                          
                          ['Spam Email Detector',
                           'Movie Recommendation',
                           'Parkinsons Prediction'],
                          icons=['envelope-exclamation','film','person'],
                          default_index=0)

# Spam Email Prediction Page
if (selected == 'Spam Email Detector'):
    
    # page title
    st.title('Spam Email Detection using ML')
    
    
    # getting the input data from the 
    SpamMail = st.text_input('Enter the Spam Email')

    # Vecotrizing the data
    
    SpamMailList = [SpamMail]
    SpamMailVectorized = vectorizer_model.transform(SpamMailList)
    
    SpamMailArray = SpamMailVectorized.toarray()
    
    # code for Prediction
    spam_mail = ''
        
    # creating a button for Prediction
        
    if st.button('Is this Email a Spam?'):
       spam_mail_prediciton = spam_model.predict(SpamMailArray)
            
       if (spam_mail_prediciton[0] == 1):
              spam_mail = 'No'
       else:
              spam_mail = 'Yes'
            
    st.success(spam_mail)

    