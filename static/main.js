document.addEventListener("DOMContentLoaded", function() {
    const contentDiv = document.querySelector(".content");

    // Handle form submission
    const form = document.querySelector("form");
    form.addEventListener("submit", function(event) {
        event.preventDefault();
        
        const formData = new FormData(form);
        fetch("/", {
            method: "POST",
            body: formData
        }).then(response => {
            if (response.ok) {
                alert("Files processed successfully.");
            } else {
                alert("Error processing files.");
            }
        });
    });

    const navButtons = document.querySelectorAll('.nav-button');
    navButtons.forEach(button => {
        button.addEventListener('click', function(event) {
            event.preventDefault();
            const url = this.getAttribute('data-url');
            loadContent(url);
        });
    });

    function loadContent(url) {
        fetch(url)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.text();
            })
            .then(data => {
                contentDiv.innerHTML = data;
                const scriptTags = contentDiv.querySelectorAll('script');
                scriptTags.forEach(scriptTag => {
                    const newScriptTag = document.createElement('script');
                    newScriptTag.src = scriptTag.src;
                    document.body.appendChild(newScriptTag);
                });
            })
            .catch(error => {
                console.error('Error fetching content:', error);
            });
    }
});
