
# Core Pkgs
import streamlit as st 
import SessionState
# EDA Pkgs
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import pytz
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal

#Preprocess
from preprocess_tweet import cleaningText, casefoldingText,tokenizingText, filteringText, stemmingText

# Naive Bayes
from NaiveBayes import NBClassifier

# Pkl_Filename = "Pickle_NB_Model.pkl"
Pkl_Filename = "pickle/nb_model.pickle" 

with open(Pkl_Filename, 'rb') as file:  
    nb = pickle.load(file)

def process_tweet(tweet):
    tweet_Cleaning = cleaningText(tweet)
    tweet_CaseFolding = casefoldingText(tweet_Cleaning)
    tweet_Tokenizing = tokenizingText(tweet_CaseFolding)
    tweet_Filtering = filteringText(tweet_Tokenizing)
    tweet_Stemming = stemmingText(tweet_Filtering)    
    return tweet_Cleaning, tweet_CaseFolding, tweet_Tokenizing, tweet_Filtering, tweet_Stemming

def process_tweet_train(tweet):
    tweet_Cleaning = cleaningText(tweet)
    tweet_CaseFolding = casefoldingText(tweet_Cleaning)
    tweet_Tokenizing = tokenizingText(tweet_CaseFolding)
    tweet_Filtering = filteringText(tweet_Tokenizing)
    # tweet_Stemming = stemmingText(tweet_Filtering)    
    return tweet_Filtering


# Fxn
def predict_sentiment(docx):
    results, post = nb.predict([docx])
    return results[0] , post

# Main Application
def main():
    st.title("Aplikasi Klasifikasi Sentimen Tweet")
    menu = ["Single Tweet","Scrap dan Klasifikasi Tweet","Training"]
    choice = st.sidebar.selectbox("Menu",menu) 
   
    if choice == "Single Tweet":

        st.subheader("Input dan Klasifikasi Tweet")

        with st.form(key='sentiment_clf_form'):
            raw_text = st.text_area("Masukan Tweet yang ingin anda klasifikasikan")
            submit_text = st.form_submit_button(label='Tekan Untuk Melakukan Klasifikasi')

        if submit_text:
            col1,col2  = st.columns(2)

            # Apply Fxn Here
            tweet_Cleaning, tweet_CaseFolding, tweet_Tokenizing, tweet_Filtering, tweet_Stemming = process_tweet(raw_text)
            prediction, post = predict_sentiment(tweet_Stemming)

            with col1:
                st.success("Teks Asli")
                st.write(raw_text)
                
                st.success("Teks Setetelah Cleaning")
                st.write(tweet_Cleaning)                
                
                st.success("Teks Setelah Casefolding")
                st.write(tweet_CaseFolding)
                
                st.success("Teks Setelah Tokenizing")
                st.write(tweet_Tokenizing)

                st.success("Teks Setelah Filtering")
                st.write(tweet_Filtering)
                
                st.success("Teks Setelah Stemming")
                st.write([tweet_Stemming])
                
                st.success("Hasil Naive Bayes")           
                     
                posterior_netral = '{0:.16f}'.format(post[0])
                st.write(f" > Posterior Probability Netral: {posterior_netral} ")
                
                posterior_positif = '{0:.16f}'.format(post[1])
                st.write(f"> Posterior Probability Positif: {posterior_positif}")
                
                posterior_negatif = '{0:.16f}'.format(post[2])
                st.write(f"> Posterior Probability Negatif: {posterior_negatif} ")

                
                if prediction == 0:
                    st.write("Berdasarkan Hasil diatas Tweet yang anda masukan memiliki sentimen Netral")
                elif prediction == 1:
                    st.write("Berdasarkan Hasil diatas Tweet yang anda masukan memiliki sentimen Positif")
                elif prediction == 2:
                    st.write("Berdasarkan Hasil diatas Tweet yang anda masukan memiliki sentimen Negatif")
                else:
                    st.write("Error!")

    elif choice == "Scrap dan Klasifikasi Tweet":
        st.subheader("Scrap Tweet Berdasarkan Kata Kunci")


        with st.form(key='tweet_test_form'):
            kata_kunci = st.text_area("Masukan Kata Kunci")
            jml_tweets = st.text_input("Masukan Jumlah Jumlah Tweets")
            
            submit_text = st.form_submit_button(label='Tekan Untuk Klasifikasi Hasil Scrap')

        if submit_text:
            import tweepy as tw

            ####Credentials
            consumer_key='VClbThxnr6T59VVrXtJJ1c7yF'
            consumer_secret='SKR1Oxo1MTgn6veEEfuEj76nrTaklhtxYP0mlFjHIneJEKubf1'
            access_token='1362975019131760642-5EjLvg3qNvIAVVa5ui9CO27RWHhWTQ'
            access_token_secret='8l1NLLiRnd4XIVDtFnoW75GrABPffKB2bR4SgDy5A72VP'
            
            # Authenticate
            auth = tw.OAuthHandler(consumer_key, consumer_secret)
            # Set Tokens
            auth.set_access_token(access_token, access_token_secret)
            # Instantiate API
            api = tw.API(auth, wait_on_rate_limit=True)
            
            #Download Tweet (Data Test)
            KataKunci = kata_kunci
            JmlTweets = int(jml_tweets)
            searched_tweets = [status for status in tw.Cursor(api.search_tweets, q=KataKunci).items(JmlTweets)]
            test_tweet = [tweet.text for tweet in searched_tweets]
            
            preprocessed_text = []            
            if test_tweet:

                for i in range(0, len(test_tweet)):
                    tweet_Cleaning, tweet_CaseFolding, tweet_Tokenizing, tweet_Filtering, tweet_Stemming = process_tweet(test_tweet[i])
                    preprocessed_text.append(tweet_Stemming)
                
                X_test_tweet = preprocessed_text
                
                tweet_clean = X_test_tweet
                
                predict_tweet = []
                
                for i in range(0, len(tweet_clean)):
                    res, pos = predict_sentiment(tweet_clean[i])
                    predict_tweet.append(res)
                
                prediksi_tweet = predict_tweet
                
                hasil = pd.DataFrame(list(zip(test_tweet, prediksi_tweet)),
                columns =['Tweet', 'Sentimen'])
                
                hasil['Sentimen'].replace((0, 1, 2), ('neutral', 'positive', 'negative'), inplace=True)
                
                st.success("Hasil Klasifikasi Sentimen")
                st.write(hasil)

                fig, ax = plt.subplots(figsize = (6, 6))
                sizes = [count for count in hasil['Sentimen'].value_counts()]
                labels = list(hasil['Sentimen'].value_counts().index)
                # explode = (0.1, 0, 0)
                ax.pie(x = sizes, labels = labels, autopct = '%1.1f%%', textprops={'fontsize': 14})
                ax.set_title('Polaritas Sentimen Data dari Tweet', fontsize = 16, pad = 20)
                # plt.show()

                st.success("Visualisasi Pie Chart")
                st.write(fig)
            else:
                st.write("Tweet Tidak ditemukan!")
            
    else:
        st.subheader("Training")
        file = st.file_uploader("Masukan Dataset (CSV Format) :", type=["csv"])
        session_state = SessionState.get(name="", file=False)


        if file:
            session_state.file = True

        if session_state.file:
            if file is None:
                st.text("Masukkan Dataset Terlebih Dahulu")
            else:
                st.subheader("Data Training")
                data_frame = pd.read_csv(file)
                data_frame = data_frame[:5000]
                # data_frame.head()

                st.write(data_frame)
                
                #nanti pindahin ke function sendiri
                data_frame['polarity'].replace(('neutral', 'positive', 'negative'), (0, 1, 2), inplace=True)

                data = data_frame['text_preprocessed'].values.tolist()
                label = data_frame['polarity'].values.tolist()

                st.subheader("Input Parameter")
                with st.form(key='Train_model'):
                    session_state.test_size = st.number_input("Parameter Test Split")
                    session_state.submit_param = st.form_submit_button(label='Tekan Untuk Melakukan Training') 
                try:
                    if session_state.submit_param:
                        train_X, test_X, y_train, y_test = train_test_split(data, label, test_size=session_state.test_size, shuffle=True)

                        st.subheader("Data Spliting")
                        st.write('Jumlah Training: ', len(train_X))
                        st.write('Jumlah Testing: ', len(test_X))
                        # st.write('Jumlah Testing: ', session_state.test_size)

                        time_jkt = pytz.timezone('Asia/Jakarta')
                        print("Mulai:", datetime.now(time_jkt).strftime("%H:%M:%S"))

                        st.subheader("Text Preprocessing")
                        with st.spinner('Proses Text Preprocessing Sedang Berjalan..'):
                        
                            #preprocess data train
                            preprocessed_text = []
                            for i in range(0, len(train_X)):
                                preprocessed_text.append(process_tweet_train(train_X[i]))

                            X_train = preprocessed_text

                            Pkl_Filename = "pickle/X_train.pkl"  

                            with open(Pkl_Filename, 'wb') as file:  
                                pickle.dump(X_train, file)

                            #preprocess data test
                            preprocessed_text = []
                            for i in range(0, len(test_X)):
                                preprocessed_text.append(process_tweet_train(test_X[i]))

                            X_test = preprocessed_text

                            Pkl_Filename = "pickle/y_train.pkl"  

                            with open(Pkl_Filename, 'wb') as file:  
                                pickle.dump(y_train, file)

                            print("Selesai:", datetime.now(time_jkt).strftime("%H:%M:%S"))

                            st.success("Text Preprocessing Selesai")
                            # st.write("Preprocessed Training Data")
                            # st.write(X_train)
                            # st.write("Preprocessed Testing Data")
                            # st.write(X_test)

                        st.subheader("Training")
                        with st.spinner('Proses Training Sedang Berjalan..'):
                            #training classifier     
                            nb = NBClassifier(X_train, y_train, 'full')  
                            nb.train()

                            f = open('pickle/nb_model.pickle', 'wb')
                            pickle.dump(nb, f)
                            f.close()

                            st.success("Training Selesai")

                        st.subheader("Testing")
                        with st.spinner('Proses Testing Sedang Berjalan..'):
                            #testing
                            Pkl_Filename = 'pickle/nb_model.pickle'
                            with open(Pkl_Filename, 'rb') as file:  
                                nb = pickle.load(file)

                            y_pred, pos = nb.predict(X_test)

                            # pd.set_option('display.max_colwidth', 3000)
                            # hasil = pd.DataFrame(list(zip(data_frame['text_preprocessed'], X_test, y_pred)),
                            #                columns =['Tweet', 'Tweet_Processed', 'Sentimen'])
                            # hasil['Sentimen'].replace((0, 1, 2), ('neutral', 'positive', 'negative'), inplace=True)

                            # st.write("Sample Testing")
                            # st.write(hasil)

                            #Evaluasi Performa Model NB Manual
                            st.write("Akurasi")

                            cor1, acc1 = nb.score(y_pred, y_test)

                            print("Prediksi Benar:", cor1)
                            print("Akurasi: %i / %i = %.4f " %(cor1, len(y_pred), acc1))

                            st.success("Akurasi: %i / %i = %.4f " %(cor1, len(y_pred), acc1))
                            
                except ValueError:
                    st.error("test_size harus positif dan lebih kecil dari jumlah sampel  atau float dalam rentang (0, 1)")

if __name__ == '__main__':
	main()
