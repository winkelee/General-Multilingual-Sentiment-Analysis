document.addEventListener("DOMContentLoaded", function(){


const sourceText = document.getElementById("sourceText");
const classifyButton = document.getElementById("classifyButton");
const predictedInfo = document.getElementById("predictedInfo");
const tokenCount = document.getElementById("tokenCount");
const tokenWarning = document.getElementById("tokenWarning");
const predictedStats = document.getElementById('predictedStats')

const API_PREDICT_ENDPOINT = "http://127.0.0.1:8000/predict/"

let inputText = null;
let inputTextFinal = null;
let classified = false;

function countWords(s){
    s = s.replace(/(^\s*)|(\s*$)/gi,"");//exclude  start and end white-space
    s = s.replace(/[ ]{2,}/gi," ");//2 or more space to 1
    s = s.replace(/\n /,"\n"); // exclude newline with a start spacing
    return s.split(' ').length; 
}

sourceText.addEventListener('keyup', (e) => {
    console.log("Taking input from the input field")
    inputText = sourceText.value;
    wordCount = countWords(inputText)
    tokenCount.textContent = `Token count: ${wordCount}/100`
    if (wordCount > 100){
        tokenWarning.textContent = `Warning! All the tokens after 100 will be discarded.`
        tokenWarning.style.color = '#710000'
    } else{
        tokenWarning.textContent = ''
    }
});

classifyButton.addEventListener('click', (e) => {
    if (sourceText.value == ''){
        clean_output()
    } else{
        console.log("Parsing input: ", sourceText.value)
        clean_output()
        predictSentiment()
    }
    
})

function clean_output(){
    const naiveSentimentLabel = document.getElementById('naiveSentimentLabel')
    const generalSentimentLabel = document.getElementById('generalSentimentLabel')
    const toxicityLabel = document.getElementById('toxicityLabel')
    const toxicityConfidenceLabel = document.getElementById('toxicityConfidenceLabel')
    const confidenceLabel = document.getElementById('confidenceLabel')
    const languageLabel = document.getElementById('languageLabel');
    const translationNeededLabel = document.getElementById('translationNeededLabel')
    const jsonRawLabel = document.getElementById('jsonRawLabel')
    if (naiveSentimentLabel) naiveSentimentLabel.remove();
    if (generalSentimentLabel) generalSentimentLabel.remove();
    if (toxicityLabel) toxicityLabel.remove();
    if (toxicityConfidenceLabel) toxicityConfidenceLabel.remove();
    if (confidenceLabel) confidenceLabel.remove();
    if (languageLabel) languageLabel.remove();
    if (translationNeededLabel) translationNeededLabel.remove();
    if (jsonRawLabel) jsonRawLabel.remove();
    classified = false
}


function predictSentiment(){
    inputTextFinal = sourceText.value
    sourceText.diabled = true
    classifyButton.textContent = "Classifying..."
    classifyButton.disabled = true
    fetch(API_PREDICT_ENDPOINT, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({source_text: inputTextFinal})
    }).then(response =>{
        if(!response.ok){
            response.json().then(error =>{
                throw new Error(error.detail || `API ERROR ${response.statusText} STATUS: ${response.status}`);
            }).catch(() => {
                throw new Error(`API ERROR: ${response.statusText} STATUS: ${response.status}`);
            })
        }
        return response.json()
    }).then(data => {
        classified = true;
        console.log("Retrieved data from API: ", data);
        const predictedLanguage = data.predicted_language;
        const predictionTranslationNeeded = data.prediction_translation_needed;
        const predictionClass = data.prediction_class;
        const confidence = data.confidence.toFixed(4);
        const toxicityPredictionClass = data.toxicity_class;
        const toxicityConfidence = data.toxicity_confidence.toFixed(4);
        const generalSentiment = data.general_sentiment;
        const rawJSON = data;
        setPredictedData(predictedLanguage, predictionTranslationNeeded, predictionClass, confidence, toxicityPredictionClass, toxicityConfidence, generalSentiment, rawJSON)
        classifyButton.textContent = "Classify!"
        classifyButton.disabled = false
        sourceText.disabled = false
    })
}

function setPredictedData(predictedLanguage, predictionTranslationNeeded, predictionClass, predictionConfidence, toxicityPredictionClass, toxicityConfidence, generalSentiment, rawJSON){
    const generalSentimentLabel = document.createElement('span')
    generalSentimentLabel.style.fontSize = '18px'
    generalSentimentLabel.id = 'generalSentimentLabel'
    generalSentimentLabel.textContent = `General sentiment: ${generalSentiment}`
    predictedStats.appendChild(generalSentimentLabel)
    const naiveSentimentLabel = document.createElement('span')
    naiveSentimentLabel.style.fontSize = '18px'
    naiveSentimentLabel.id = 'naiveSentimentLabel'
    naiveSentimentLabel.textContent = `Naive sentiment: ${predictionClass}`
    predictedStats.appendChild(naiveSentimentLabel)
    //const confidenceLabel = document.createElement('span')
    //confidenceLabel.style.fontSize = '18px'
    //confidenceLabel.id = 'confidenceLabel'
    //confidenceLabel.textContent = `Naive confidence: ${predictionConfidence}`
    //predictedStats.appendChild(confidenceLabel)
    const toxicityLabel = document.createElement('span')
    toxicityLabel.id = 'toxicityLabel'
    toxicityLabel.style.fontSize = '18px'
    toxicityLabel.textContent = `Toxicity sentiment: ${toxicityPredictionClass}`
    predictedStats.appendChild(toxicityLabel)
    //const toxicityConfidenceLabel = document.createElement('span')
    //toxicityConfidenceLabel.id = 'toxicityConfidenceLabel'
    //toxicityConfidenceLabel.style.fontSize = '18px'
    //toxicityConfidenceLabel.textContent = `Toxicity confidence: ${toxicityConfidence}`
    //predictedStats.appendChild(toxicityConfidenceLabel)
    const languageLabel = document.createElement('span');
    languageLabel.style.fontSize = '18px'
    languageLabel.id = 'languageLabel'
    languageLabel.textContent = `Predicted input language: ${predictedLanguage}`
    predictedStats.appendChild(languageLabel)
    //const translationNeededLabel = document.createElement('span')
    //translationNeededLabel.style.fontSize = '18px'
    //translationNeededLabel.id = 'translationNeededLabel'
    //translationNeededLabel.textContent = `Input translated: ${predictionTranslationNeeded}`
    //predictedStats.appendChild(translationNeededLabel);
    const jsonRawLabel = document.createElement('span')
    jsonRawLabel.style.fontSize = '18px'
    jsonRawLabel.id = 'jsonRawLabel'
    jsonRawLabel.textContent = 'View raw JSON output...'
    jsonRawLabel.style.cursor = 'pointer'
    jsonRawLabel.style.textDecoration = 'underline'
    jsonRawLabel.addEventListener('click', () => {
        alert(JSON.stringify(rawJSON, null, 2));
    })
    predictedInfo.appendChild(jsonRawLabel)
}

})