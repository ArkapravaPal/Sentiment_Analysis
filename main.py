import streamlit as st
import pickle
from utils import process_review
from PIL import Image

# values of logprior & loglikelihood generated from training corpus
logprior = -0.0039000049432615924
with open('loglikelihood.pickle', 'rb') as handle:
    loglikelihood = pickle.load(handle)

def nb_predict(review, logprior, loglikelihood):
    '''
    Input:
        review: a string
        logprior: a number
        loglikelihood: a dictionary of words mapping to numbers
    Output:
        p: the sum of all the logliklihoods of each word in the review (if found in the dictionary) + logprior

    '''
    
    # process the review to get a list of words
    word_l = process_review(review)

    p = 0
    p += logprior

    for word in word_l:
        # check if the word exists in the loglikelihood dictionary
        if word in loglikelihood:
            # add the log likelihood of that word 
            p += loglikelihood[word]

    
    if p>0:
        return "Positive"
    else:
        return "Negative"

########################################## streamlit work #########################################

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
		prediction = nb_predict(user_input, logprior, loglikelihood)
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