function fetchKeywords() {
   console.log("2fetevndgsfdcwvrgf")
    let url = document.getElementById("urlInput").value;
    let result = document.getElementById("result")
    if (!url) {
        alert("Please enter a product URL.");
        return;
    }

    fetch('/extract', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ url: url })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data)
        if (!data) {
            document.getElementById("result").innerHTML = `<p style="color:red;">${data.error}</p>`;
        } else {
            console.log(data,"data")
            result.innerHTML = `
                <h3>Extracted Product Name:</h3><p>${data.product_name}</p>
                <h3>Common Words:</h3><p>${data.common_words.join(", ")}</p>
                <h3>Optimized Keywords:</h3><p>${data.optimized_keywords.join(", ")}</p>
                <h3>Meaningful Keyword Combinations:</h3><p>${data.meaningful_keywords.join(", ")}</p>
            `;
        }
    })
    // .catch(error => console.log("Error fetching data:", error));
}
