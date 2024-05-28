document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('TClustAE').addEventListener('click', function(event) {
        event.preventDefault(); // Prevent default form submission
        document.getElementById('hidden_model').value = 'TClustAE';
        document.getElementById('form_model').submit();
        sessionStorage.setItem('hidden_model', 'TClustAE');
        //localStorage.setItem("model", document.getElementById("model").innerText);
        //showForms('TClustAE');         
    });

    document.getElementById('STClustAE').addEventListener('click', function(event) {
        event.preventDefault(); // Prevent default form submission
        document.getElementById('hidden_model').value = 'STClustAE';
        document.getElementById('form_model').submit(); 
        //localStorage.setItem("model", document.getElementById("model").innerText);
        sessionStorage.setItem('hidden_model', 'STClustAE');
        //showForms('STClustAE');
        
    });
});
