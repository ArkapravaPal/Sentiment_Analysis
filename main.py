import streamlit as st
import pickle
from utils import process_tweet
# import nltk
from PIL import Image

logprior = -0.0039000049432615924
with open('loglikelihood.pickle', 'rb') as handle:
    loglikelihood = pickle.load(handle)

def naive_bayes_predict(tweet, logprior, loglikelihood):
    '''
    Input:
        tweet: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p: the sum of all the logliklihoods of each word in the tweet (if found in the dictionary) + logprior (a number)

    '''
    ### START CODE HERE ###
    # process the tweet to get a list of words
    word_l = process_tweet(tweet)

    # initialize probability to zero
    p = 0

    # add the logprior
    p += logprior

    for word in word_l:

        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word to the probability
            p += loglikelihood[word]

    ### END CODE HERE ###
    if p>0:
        return "Positive"
    else:
        return "Negative"

# def predict_on_ip(usr_ip):
# 	usr_ip = text_cleaner(usr_ip)
# 	review_vec = count_vect.transform([usr_ip])
# 	result = {1:"Positive",0:"Negative"}
# 	return result[model.predict(review_vec)[0]]


# model = joblib.load("trained_model.pkl")
# count_vect = joblib.load("vectors.pkl")



nav= st.sidebar.radio("Navigations",["Home","Data","Model","Code","Contact Us"],index=0)

st.sidebar.write("""Contact us \n
pal.arkaprava@gmail.com \n
+91-9932531127""")

if nav == "Home":
	st.title('Movie Review Sentiment Analysis!!')
	img = Image.open(r"review image.png")
	st.image(img)
	st.header("""Write your review about the movie and predict the Sentiment""")
	user_input = st.text_area("Please enter your movie review here")
	def predict():
		if len(user_input) == 0:
			return st.error("Please enter a valid review")
		prediction = naive_bayes_predict(user_input, logprior, loglikelihood)
		st.write("Your review's sentiment is : ",prediction)
		return
	if st.button("Predict"):
		predict()


if nav == "Data":
	st.title("Read more about Data")
	st.write("## Data Source","Kaggle.com")

if nav == "Model":
	st.header("Model Training")

if nav == "Code":
	st.title("Python code for model training")

if nav == "Contact Us":
	st.title("Welcome to the world of Predictions")
	st.subheader("Please feel free to write us about your experience with us")