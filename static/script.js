function showMessage(message,type = 'info') {
    const container = document.getElementById('message-container');
    if (!container) {
        console.error('Error : Could not find the div with id message-containe');
        return
    }

    const messageBox = document.createElement('div');
    messageBox.textContent = message;

     let messageClass = '';

    if (type === 'error') {
        messageClass = 'text-red-600';
    } else if (type === 'success') {
        messageClass = 'text-green-600';
    } else {
        messageClass = 'text-gray-600';
    }

    messageBox.className = `p-3 rounded-lg text-sm font-medium ${messageClass}`;
    container.innerHTML = '';
    container.appendChild(messageBox);

    setTimeout(()=>{
        messageBox.remove();
    },3000);
}

function drawWordCloud(words) {
            const canvas = document.getElementById('word-cloud-canvas');
            const label = document.getElementById('word-cloud-label');
            
            if (typeof WordCloud !== 'undefined') {
                if (!words || words.length === 0) {
                    WordCloud(canvas, { list: [] });
                    label.textContent = 'No words to display for this topic.';
                } else {
                    label.textContent = 'Most important words for this topic:';
                    
                    const formattedWords = words.map(word_and_prob => {
                        const [word,probability] = word_and_prob;
                        const size = probability * 500;
                        return [word,size]
                    })
                    

                    WordCloud(canvas, {
                        list: formattedWords,
                        shuffle: false,
                        minFontSize: 10,
                        maxFontSize: 80,
                        backgroundColor: '#F9FAFB',
                        color: 'random-dark'
                    });
                }
            } else {
                label.textContent = 'Word cloud library is not loaded.';
                console.error('WordCloud library not found. Please check the CDN link.');
            }
        }

document.addEventListener('DOMContentLoaded',()=>{
    const classifyBtn = document.getElementById('classify-btn');
    const textInput = document.getElementById('text-input');
    const topicName = document.getElementById('topic-name');
    const confidence = document.getElementById('confidence');

    classifyBtn.addEventListener('click', async () => {
        const text = textInput.value;
        if (!text) {
            alert('Please entre some text to classify');
            return;
        }

        classifyBtn.textContent = 'Classifiying...';
        classifyBtn.disabled = true;
        topicName.textContent = '....';
        confidence.textContent = '...';
        drawWordCloud([]);

        try {
            const response = await fetch('http://localhost:5000/predict', {
                method : 'POST',
                headers : {
                    'Content-Type' : 'application/json'
                },
                body : JSON.stringify({ text : text })

            });

            if(!response.ok) {
                throw new Error(`HTTP error! status : ${response.status}`);
            }
            const data = await response.json();
            if (data.message) {
                topicName.textContent = 'N/A';
                confidence.textContent = '0.00%';
                showMessage(data.message, 'info');
                drawWordCloud([]);

            } else if (data.error) {
                showMessage(`Error from server: ${data.error}`, 'error');
                topicName.textContent = 'Error';
                confidence.textContent = 'Could not classify topic.';
                drawWordCloud([]);
            } else {
                topicName.textContent = data.predicted_topic;
                confidence.textContent = `${ (data.confidence * 100).toFixed(2) }%`;
                drawWordCloud(data.top_words);
                showMessage('Classification successful!', 'success');
            }    
        } catch(error){
            console.error('Error during classification', error)
            topicName.textContent = 'Error';
            confidence.textContent = 'could not classify topic.'
        } finally {
            classifyBtn.textContent = 'Classify Topic';
            classifyBtn.disabled = false;
        }

    });
});