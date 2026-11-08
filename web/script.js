const button = document.getElementById("do-stuff")
const imageUpload = document.getElementById("image-file-input")
const annotation = document.getElementById("annotation-file-input")
const imageResult = document.getElementById("image-result")
const imageContainer = document.getElementById("image-container")

let imageLoaded = null;
let jsonLoaded = null;

const imageReader = new FileReader();
imageReader.onload =  (event)=>{
    imageLoaded = event.target.result;
    if (jsonLoaded!=null) sendData();
};

const jsonReader = new FileReader();
jsonReader.onload = (event) => {
    jsonLoaded = event.target.result;
    if (imageLoaded!=null) sendData();
}

function drawBox(result) {
    const box = document.createElement("div");
    box.classList.add("box")
    box.style["height"] = "100px"
    box.style["z-index"] = "1"
    imageContainer.appendChild(box)

}

function sendData(){
    console.log(imageLoaded);
    console.log(jsonLoaded);

    const xhr = new XMLHttpRequest();
    xhr.open("POST","/api/thing");
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.send(JSON.stringify({
        annotation: jsonLoaded,
        image: imageLoaded
    }))
    xhr.onreadystatechange = () => {
        console.log(xhr.readyState)
       if (xhr.readyState !== 4) return;
        if (xhr.status === 200) {
            const data = JSON.parse(xhr.responseText);
            console.log(data["image"])
            imageResult.src="data:image/jpeg;base64, "+data["image"]
            // we get the returned data
            drawBox(data["result"])
        }

    }

    imageLoaded=null;
    jsonLoaded=null;

    button.disabled=false
}

button.onclick = () => {
    button.disabled = true

    imageReader.readAsDataURL(imageUpload.files[0])
    jsonReader.readAsText(annotation.files[0])
}
