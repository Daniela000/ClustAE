// Function to get query parameters from URL
function getQueryParams() {
    const params = {};
    const queryString = window.location.search;
    const urlParams = new URLSearchParams(queryString);

    urlParams.forEach((value, key) => {
        params[key] = value;
    });

    return params;
}


// Extract the variable from the URL
const params = getQueryParams();
const yourVariable = params['model']; // Change 'model' to the actual variable name if different

// Add the variable to the form's action URL
const submitButton = document.getElementById('submit_questions');
submitButton.addEventListener('click', function(event) {
    event.preventDefault(); // Prevent default form submission

    // Get the form element
    const form = document.getElementById('form_questions');
    // Get the current form action URL
    let actionUrl = form.action;
    // Add the extracted variable to the action URL
    if (yourVariable) {
        actionUrl += `?model=${yourVariable}`;
    }
    // Set the form's action to the new URL
    form.action = actionUrl;


    const formData = new FormData(form);

    // Send the data to the Flask backend
    fetch(form.action, {
        method: 'POST',
        body: formData
    });

    // Submit the form
    form.submit();
});

