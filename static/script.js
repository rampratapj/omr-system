// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    loadAnswerKeys();
    loadResults();
    setupDragDrop();
});

// Setup drag and drop
function setupDragDrop() {
    const fileInput = document.getElementById('omrFile');
    const dropZone = document.querySelector('.file-input-label');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.style.background = '#e8e8ff';
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.style.background = '#f5f5f5';
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        fileInput.files = files;
    });
}

// Load answer keys
function loadAnswerKeys() {
    fetch('/api/answer-keys')
        .then(r => r.json())
        .then(data => {
            const select = document.getElementById('answerKey');
            select.innerHTML = '<option value="">-- Choose Answer Key --</option>';
            
            data.keys.forEach(key => {
                const option = document.createElement('option');
                option.value = key.name;
                option.textContent = `${key.name} (${key.questions} Q)`;
                select.appendChild(option);
            });

            updateKeysList(data.keys);
        })
        .catch(err => showAlert('keyAlert', 'Error: ' + err, 'error'));
}

// Update keys list
function updateKeysList(keys) {
    const list = document.getElementById('keysList');
    if (keys.length === 0) {
        list.innerHTML = '<p style="color: #999;">No keys created</p>';
        return;
    }
    list.innerHTML = keys.map(key => 
        `<div style="padding: 8px 12px; background: white; margin: 5px 0; border-left: 4px solid #667eea; border-radius: 4px;">
            ✓ ${key.name} (${key.questions} questions)
            <button onclick="deleteKey('${key.name}')" style="float: right; background: #e74c3c; padding: 4px 8px; font-size: 0.8em; width: auto;">Delete</button>
        </div>`
    ).join('');
}

// Delete answer key
function deleteKey(keyName) {
    if (!confirm(`Delete key: ${keyName}?`)) return;
    
    fetch(`/api/delete-key/${keyName}`, {method: 'DELETE'})
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                loadAnswerKeys();
                showAlert('keyAlert', 'Key deleted', 'success');
            } else {
                showAlert('keyAlert', data.message, 'error');
            }
        });
}

// Create answer key
function createAnswerKey() {
    const keyName = document.getElementById('keyName').value.trim();
    const answersText = document.getElementById('answers').value.trim();

    if (!keyName) {
        showAlert('keyAlert', 'Enter key name', 'error');
        return;
    }

    if (!answersText) {
        showAlert('keyAlert', 'Enter answers', 'error');
        return;
    }

    const answers = answersText.split(',').map(a => {
        const num = parseInt(a.trim());
        return isNaN(num) ? -1 : num;
    });

    fetch('/api/create-answer-key', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({key_name: keyName, answers: answers})
    })
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                showAlert('keyAlert', 'Key created!', 'success');
                document.getElementById('keyName').value = '';
                document.getElementById('answers').value = '';
                loadAnswerKeys();
            } else {
                showAlert('keyAlert', data.message, 'error');
            }
        })
        .catch(err => showAlert('keyAlert', 'Error: ' + err, 'error'));
}

// Upload OMR
function uploadOMR() {
    const file = document.getElementById('omrFile').files[0];
    const answerKey = document.getElementById('answerKey').value;
    const rollNumber = document.getElementById('rollNumber').value || 'Unknown';
    const studentName = document.getElementById('studentName').value || 'N/A';

    if (!file) {
        showAlert('uploadAlert', 'Select OMR file', 'error');
        return;
    }

    if (!answerKey) {
        showAlert('uploadAlert', 'Select answer key', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('answer_key', answerKey);
    formData.append('roll_number', rollNumber);
    formData.append('student_name', studentName);

    showSpinner(true);

    fetch('/api/upload', {method: 'POST', body: formData})
        .then(r => r.json())
        .then(data => {
            showSpinner(false);
            if (data.success) {
                displayResult(data.result, data.chart, data.student_name, data.roll_number);
                showAlert('uploadAlert', 'OMR processed!', 'success');
                loadResults();
            } else {
                let msg = data.message;
                if (data.issues) msg += ': ' + data.issues.join(', ');
                showAlert('uploadAlert', msg, 'error');
            }
        })
        .catch(err => {
            showSpinner(false);
            showAlert('uploadAlert', 'Error: ' + err, 'error');
        });
}

// Display result
function displayResult(result, chart, name, roll) {
    document.getElementById('resultName').textContent = name;
    document.getElementById('resultRoll').textContent = roll;
    document.getElementById('resultTotal').textContent = result.total;
    document.getElementById('resultCorrect').textContent = result.correct;
    document.getElementById('resultIncorrect').textContent = result.incorrect;
    document.getElementById('resultSkipped').textContent = result.skipped;
    document.getElementById('finalScore').textContent = result.score.toFixed(2) + '%';
    document.getElementById('resultChart').src = 'data:image/png;base64,' + chart;
    document.getElementById('resultSection').style.display = 'block';
}

// Load results
function loadResults() {
    fetch('/api/results')
        .then(r => r.json())
        .then(data => {
            const tbody = document.getElementById('resultsTable');
            if (!data.results || data.results.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: #999;">No results</td></tr>';
                return;
            }

            tbody.innerHTML = data.results.map(r => `
                <tr>
                    <td>${r.roll_number}</td>
                    <td>${r.name}</td>
                    <td>${r.total}</td>
                    <td><span class="badge success">${r.correct}</span></td>
                    <td><span class="badge error">${r.incorrect}</span></td>
                    <td><span class="badge warning">${r.skipped}</span></td>
                    <td><strong>${r.score.toFixed(2)}%</strong></td>
                    <td>${new Date(r.timestamp).toLocaleString()}</td>
                </tr>
            `).join('');
        })
        .catch(err => console.error('Error:', err));
}

// Load statistics
function loadStatistics() {
    fetch('/api/statistics')
        .then(r => r.json())
        .then(data => {
            document.getElementById('statTotal').textContent = data.total_evaluated;
            document.getElementById('statAvg').textContent = data.avg_score.toFixed(2) + '%';
            document.getElementById('statMax').textContent = data.max_score.toFixed(2) + '%';
            document.getElementById('statsContainer').style.display = 'block';
        })
        .catch(err => console.error('Error:', err));
}

// Export results
function exportResults() {
    window.location.href = '/api/export-results';
}

// Utilities
function showAlert(elementId, message, type) {
    const alert = document.getElementById(elementId);
    alert.textContent = message;
    alert.className = 'alert ' + type;
    if (type === 'success') {
        setTimeout(() => alert.className = 'alert', 5000);
    }
}

function showSpinner(show) {
    document.getElementById('uploadSpinner').style.display = show ? 'block' : 'none';
}
