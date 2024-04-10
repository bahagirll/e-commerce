import streamlit as st
st.title("Monitoring of User Behavior of an eCommerce site")
st.sidebar.title("Table of contents")
pages=["Project overview","Pre-Processing", "DataVizualization","Feature Engineering and Modelling","Conclusion"]
page=st.sidebar.radio("Go to", pages)
# Names of the creators
creators = ["Fruzsina", "Valentin", "Ranya", "Julian"]
@st.cache_data
# Display the names in the sidebar
st.sidebar.write("Created by:")   
for creator in creators:
    st.sidebar.write(creator)
st.sidebar.write('2024 February-April  Data Analyst Bootcamp')

import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

base="dark"
secondaryBackgroundColor="#f2f8fc"
font="Helvetica"
event_df = pd.read_csv('events.csv')
category_tree = pd.read_csv('category_tree.csv')
item_properties_part1 = pd.read_csv('item_properties_part1.csv')
item_properties_part2 = pd.read_csv('item_properties_part2.csv')
if page == pages [0] : 
   st.markdown(
    """
    <div style="background-color:#ffffff; padding:10px;">
        <img src="ConversionPredictionModel.jpg" style="width:100%;" />
        <h1 style="color:#001f3f; font-size:16pt;">Conversion Prediction Model</h1>
        <h2 style="color:#001f3f; font-size:14pt;">Helping Ecommerce Operators improving their conversion rate</h2>
        <p style="font-size:12pt;">
            When users engage with an e-commerce site, various information in regards to their online activity is tracked
            and stored as data. This data can be used to understand user behavior on an e-commerce site and help building models.
            Specifically, the focus lies on developing a Conversion Prediction Model. This model endeavors to discern patterns
            within the dataset that can accurately identify which visitors will likely proceed with a transaction. By leveraging
            advanced machine learning algorithms and predictive analytics, the aim is to forecast visitor behavior and anticipate
            conversion events with precision. This predictive capability holds immense potential for e-commerce operators,
            enabling them to tailor marketing strategies, optimize resource allocation, optimize website design, and enhance
            the overall efficiency of their platforms.
        </p>
        <h2 style="color:#001f3f; font-size:14pt;">Unlocking E-commerce Success: The Power of Conversion Prediction Models</h2>
        <p style="font-size:12pt;">
            Creating a conversion prediction model for e-commerce operators is vital for maximizing sales numbers and optimizing
            business operations. Firstly, such models enable operators to allocate marketing resources more effectively by
            identifying the channels and campaigns likely to yield the highest conversions. This ensures a higher return on
            investment (ROI) and better utilization of resources.
            Secondly, prediction models facilitate personalized customer experiences. By understanding each customer's likelihood
            to convert, operators can tailor product recommendations, promotions, and messaging accordingly. This personalization
            increases the probability of successful conversions and fosters customer loyalty.
            Additionally, these models aid in inventory management by forecasting demand for specific products. By maintaining
            optimal stock levels, e-commerce operators can avoid stockouts and excess inventory, capitalizing on potential sales
            opportunities and enhancing customer satisfaction.
            Moreover, prediction models contribute to a smoother user experience by streamlining the checkout process based on
            past behavior and preferences. This reduces cart abandonment rates and increases conversion rates, resulting in
            improved overall customer satisfaction and retention.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

if page == pages[1] : 
  st.write("### Presentation of data")
  st.dataframe(event_df.head(10))
  st.dataframe(category_tree.head(10))
  st.dataframe(item_properties_part1.head(10))
  st.dataframe(item_properties_part2.head(10))
  st.write("""
    In total, there are 4 datasets available, which were taken from the website: [Kaggle RetailRocket eCommerce Dataset](https://www.kaggle.com/retailrocket/ecommerce-dataset/home) and are allowed to be shared and modified under the license: CC BY-NC-SA 4.0 DEED.
    The item property datasets were used in the Exploratory Analysis for visualizations concerning the price. The category tree dataset was not used as the link between category ID to parent ID did not provide any additional value for the project objective. The main dataset that was used was the events dataset, as it pertained to the user behavior and also provided information about item transactions.

    **Data Frame**
    The dataset “events.csv” comprises 2.7 million rows of data documenting events generated by users. Each row contains the following fields:
    - Event: Denotes the type of event that occurred. Events could include various user interactions.
    - Timestamp: Records the precise date and time when the event occurred. The timestamp provides granularity, allowing analysis of user behavior over time.
    - Visitor ID: Each user or visitor to the platform is assigned a unique identifier. This ID tracks individual user interactions and analyzes their behavior patterns across different events.
    - Item ID: Identifies the specific product or item associated with the event. For instance, if the event is a purchase, the item ID would correspond to the product bought by the user.
    - Transaction ID: Applicable only for events related to transactions, such as purchases. It records a unique identifier for each transaction, facilitating analysis of purchase behavior and revenue generation.
    - Type of Event: Categorizes the event into different types, providing insights into the nature of user interactions.
    - Date of Event: Complements the timestamp by providing the date (without the time component) on which the event occurred.
    - Visitor ID Creating the Event: Similar to the Visitor ID, this field records the unique identifier of the user who initiated the event. It helps understand user engagement and the role of specific users in driving platform activity.
    - Product ID Associated with the Event: Mirrors the Item ID and specifies the product or item involved in the event.
    """)



if page == pages[1] : 
  st.write("### DataVizualization")
  fig=plt.figure()
  sns.countplot(x='event', data=event_df)
  st.pyplot(fig)

import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
def create_dataframe(visitor_list, event_df):
    array_for_df = []

    for index in visitor_list:
        v_df = event_df[event_df['visitorid'] == index]
        temp = [index]

        num_items_viewed = v_df[v_df['event'] == 'view']['itemid'].nunique()
        view_count = v_df[v_df['event'] == 'view'].shape[0]

        num_items_added = v_df[v_df['event'] == 'addtocart']['itemid'].nunique()
        items_added = v_df[v_df['event'] == 'addtocart'].shape[0]

        bought_count = v_df[v_df['event'] == 'transaction'].shape[0]
        purchased = 1 if bought_count > 0 else 0

        array_for_df.append([index, num_items_viewed, view_count, num_items_added, items_added, bought_count, purchased])

    return pd.DataFrame(array_for_df, columns=['visitorid', 'num_items_viewed', 'view_count', 'num_items_added', 'items_added', 'bought_count', 'purchased'])


if page == pages[3]:
    st.write("### Feature Engineering and Modelling")
    visitors = event_df.visitorid.unique()
    windowShoppers = event_df[event_df['event'] == "addtocart"].visitorid.unique()
    buyers = event_df[event_df.transactionid.notnull()].visitorid.unique()

    # Sort arrays by visitorID
    visitors.sort()
    windowShoppers.sort()
    buyers.sort()

    # Amount of visitors, without transaction
    visit_only = list(set(visitors) - set(buyers))

    # Amount of visitors, without addtocart
    addtocart_only = list(set(visitors) - set(windowShoppers))

    def create_dataframe(visitor_list):
        array_for_df = []

        for index in visitor_list:
            # Filter the DataFrame for the current visitor
            v_df = event_df[event_df['visitorid'] == index]

            # Initialize a list to store the data for the current visitor
            temp = [index]

            # Add the total number of unique products viewed
            num_items_viewed = v_df[v_df['event'] == 'view']['itemid'].nunique()
            temp.append(num_items_viewed)

            # Add the total number of views regardless of product type
            view_count = v_df[v_df['event'] == 'view'].shape[0]
            temp.append(view_count)

            # Add the total number of unique products added to cart
            num_items_added = v_df[v_df['event'] == 'addtocart']['itemid'].nunique()
            temp.append(num_items_added)

            # Add the total number of added to cart regardless of product type
            items_added = v_df[v_df['event'] == 'addtocart'].shape[0]
            temp.append(items_added)

            # Add the total number of purchases
            bought_count = v_df[v_df['event'] == 'transaction'].shape[0]
            temp.append(bought_count)

            # Add 1 if the visitor made a purchase, otherwise add 0
            purchased = 1 if bought_count > 0 else 0
            temp.append(purchased)

            # Append the data for the current visitor to the main list
            array_for_df.append(temp)

        # Create a DataFrame from the list of visitor data
        return pd.DataFrame(array_for_df, columns=['visitorid', 'num_items_viewed', 'view_count', 'num_items_added', 'items_added', 'bought_count', 'purchased'])

    buying_visitors_df = create_dataframe(buyers)
    st.dataframe(buying_visitors_df.head())
    st.write("To have a 75% - 25% split for the training and test data we need a total sample size of 46.876 (35.157 + 11719). To achieve this size we take the visitor only list.")

    # To avoid a bias, the list will be shuffled before the data frame creation
    random.shuffle(visit_only)

    visit_only_df = create_dataframe(visit_only[:35157])

    st.dataframe(visit_only_df.head())

    st.write("# Merge of the two data frames")
    model_df = pd.concat([buying_visitors_df, visit_only_df], ignore_index=True)
    st.dataframe(model_df.head())

    sns.pairplot(model_df, x_vars=['num_items_viewed', 'view_count', 'num_items_added', 'items_added', 'bought_count'],
                 y_vars=['num_items_viewed', 'view_count', 'num_items_added', 'items_added', 'bought_count'], hue='purchased')
    st.pyplot()

    X = model_df.drop(['purchased', 'visitorid', 'bought_count'], axis=1)
    y = model_df.purchased

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

    def prediction(classifier):
        if classifier == 'Logistic Regression':
            clf = LogisticRegression()
        elif classifier == 'Decision Tree':
            clf = DecisionTreeClassifier()
        elif classifier == 'Random Forest':
            clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        return clf

    def scores(clf, choice):
        if choice == 'Accuracy':
            return clf.score(X_test, y_test)
        elif choice == 'Confusion matrix':
            return confusion_matrix(y_test, clf.predict(X_test))

    choice = ['Logistic Regression', 'Decision Tree', 'Random Forest']
    option = st.selectbox('Choice of the model', choice)
    st.write('The chosen model is :', option)

    clf = prediction(option)
    display = st.radio('What do you want to show ?', ('Accuracy', 'Confusion matrix'))
    if display == 'Accuracy':
        st.write(scores(clf, display))
    elif display == 'Confusion matrix':
        st.dataframe(scores(clf, display))


if page == pages[4] : 
  st.write("### Conclusion")
  st.write("Here's the conclusion of our analysis.")
    
    # Insert the image using its URL
  image_path = r"C:\Users\Bahag\OneDrive\Desktop\DATA\conclusion.jpg"
  st.image(image_path, caption='Conclusion Image', use_column_width=True)
  st.write("""Limitation

While the project achieved its objectives and provided valuable insights into user behavior on the e-commerce platform, several limitations should be acknowledged:

           Data Limitations: The available data constrained the analysis, which covered a specific timeframe. Insights drawn from this period may not fully represent long-term trends or seasonal variations in user behavior. 

           Imbalanced Data: The dataset exhibited an inherent class imbalance, with a significantly higher number of non–transaction events (view) than transaction events. Addressing this class imbalance during model training may affect the model’s performance and generalization. 

           Scope of Variables: The predictive models relied on limited variables derived from user events and interactions. Additional data, such as demographic information or user preferences, could enhance the accuracy and granularity of the models.

           Temporal Dynamics: The analysis focused on understanding user behavior within the provided timeframe. The analysis did not explicitly account for changes in market dynamics, consumer preferences, or platform features over time. 
        
           Model Interpretability: While the chosen machine learning models demonstrated high predictive accuracy, their interpretability may be limited. Understanding the underlying mechanisms driving predictions and user behavior may require additional techniques or model architectures. 

           Generalization: The models developed in this project are specific to the dataset and context of the e-commerce platform studied. Generalizing the findings to other platforms or industries may require further validation and adaption of the models. 

           Abnormal User Detection: The analysis did not explicitly address the identification and handling of abnormal user behavior, including fraudulent activities. Such abnormal users may distort patterns and trends in the data, leading to biased model predictions and inaccurate insights. 
                                                      

Outlook
  
Users tend to be price sensitive according to the type of product and the price range. The model used in this project could be even more realistic if the variable price is taken into account. Especially finding a correlation between price fluctuations and user demand could be an interesting avenue to explore in the future. For example, what kind of impact do discounts have on user behavior? This presumes that the data available is not in a hashed format.

E-Commerce sites increasingly face cyber security concerns, including bot/spam traffic. This can have a negative impact on the integrity of internal data. Therefore, it is critical for companies to look for solutions on how to filter out this particular traffic from their data, as it does not add any sort of value. The abnormal traffic can be defined in many ways, though it is important to have an actual definition in order to find actionable counter-measurements then. For this project, a logical next step would be to look into the abnormal users, visualize the quantiles for user views per day in a boxplot, and then define a threshold for abnormality. After, the users above the threshold can be filtered out and run through the model again in order to analyze the differences between the model results with and without the spam traffic. 

Lastly, when adding additional data into the datasets from other marketing touchpoints it would give a more complete picture of the customer journey and would paint a picture that would be closer to a real-life scenario. To give a better example, imagine the following scenario: User A generates one view on the website, whereas User B generates 2 views. In general, without having any additional information one would assume that User B is more engaged and therefore more likely to convert. However, what if User A got to the website by typing in the website in their search bar, whereas User B accidentally clicked on a display banner on a news site? Now the story has changed, and the intent is more clear. 
""")
