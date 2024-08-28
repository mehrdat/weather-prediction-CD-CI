# Simple Weather Prediction Model


<p align="center">
<img src="./src/image1.jpg" alt="Climate Change Illustration" width="400" />
</p>
Climate change is one of the most pressing issues of our time. It's not just a problem for the future; it's happening right now. To effectively combat climate change, we first need to understand the current state of our climate and weather patterns.

I've dedicated time to studying climate change because I believe it's our responsibility to take action. The first step is understanding the present situation. Climate and weather research is a vast field, and I've started by learning the fundamentals.

## Building and Using a Simple Model

To begin my journey, I built a simple model to analyze climate data. I trained the data using a LightGBM model, which is a powerful tool but has limitations in capturing the full complexity of time series data. To improve the model’s accuracy, I may need to explore more sophisticated models like RNNs or LSTMs.

Here's a comparison of different models I tested:

| Model                       | Performance (MAE) | R² Score | Time Taken |
|-----------------------------|-------------------|----------|------------|
| Extra Trees Regressor        | 0.0383            | 0.9995   | 0.5690     |
| Random Forest Regressor      | 0.0417            | 0.9992   | 0.8460     |
| Gradient Boosting Regressor  | 0.0860            | 0.9991   | 0.7470     |
| LightGBM                     | 0.0653            | 0.9983   | 1.2540     |

## Why I Chose LightGBM
<p align="center">
<img src="./src/image2.jpg" alt="Climate Change Illustration" width="400" />
</p>
LightGBM wasn’t the fastest model, but it was easier to manage on GitHub due to its smaller file size. Other models were quicker but had much larger files, making them harder to handle on GitHub. Despite not being the fastest, LightGBM offered a good balance between performance and manageability, especially considering its strong R² score compared to the top models.

As I continue to refine my approach, I plan to explore more advanced models to improve the accuracy and reliability of climate predictions. This journey is just beginning, but every small step counts towards understanding and combating climate change.

## How the Model Works

I developed an API using Flask to allow users to input weather-related data and receive temperature predictions. This API is backed by a LightGBM model trained on various weather attributes. Here’s a brief overview of the process:

- **Data Preparation**: I cleaned and prepared the data, focusing on key variables such as wet-bulb temperature (wetb), dew point (dewpt), relative humidity (rhum), year, and month.
- **Model Training**: The data was split into training and testing sets. I used LightGBM, which is well-suited for handling large datasets and provides good performance with manageable file sizes.
- **API Development**: I created an API that takes input variables like `{"wetb": 2.2, "dewpt": -2.1, "rhum": 62.0, "year": 2020.0, "month": 5.0}` and returns temperature predictions.
- **Deployment**: The model and API were containerized using Docker, ensuring that they can be easily deployed and scaled. I also set up an evaluation system to monitor model performance continuously.

### Variables Explanation

- **Wet-bulb temperature (wetb)**: The lowest temperature that can be reached by evaporating water into the air at constant pressure.
- **Dew point (dewpt)**: The temperature at which air becomes saturated with moisture and dew can form.
- **Relative Humidity (rhum)**: The percentage of moisture in the air relative to the maximum amount of moisture the air can hold at the same temperature.
- **Year and Month**: Temporal variables used to capture seasonal and yearly patterns in the climate data.
