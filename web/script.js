const button = document.getElementById("do-stuff")
const imageFileInput = document.getElementById("image-file-input")
const annotation = document.getElementById("annotation-file-input")
const imageResult = document.getElementById("image-result")
const imageContainer = document.getElementById("image-container")
const floaty = document.getElementById("floaty")
const lithology = document.getElementById("lithology")
const topDepth = document.getElementById("top-depth")
const depth = document.getElementById("depth")
const topY = document.getElementById("top-y")
const yHeight = document.getElementById("y-height")
const imagePreviewContainer = document.getElementById("image-preview-container")
const imageUpload = document.getElementById("image-upload")
const annotationUpload = document.getElementById("annotation-upload")
const previewImageFilename = document.getElementById("preview-image-filename")
const annotationPreviewContainer = document.getElementById("annotation-preview-container")
const previewAnnotationFilename = document.getElementById("preview-annotation-filename")
const legend = document.getElementById("legend")
const legendEmpty = document.getElementById("legend-empty")
const reset = document.getElementById("reset")

reset.onclick=()=>{
    imageFileInput.value=""
    annotation.value=""
    imageFileInput.onchange()
    annotation.onchange()
}

let imageLoaded = null;
let jsonLoaded = null;

const imageReader = new FileReader();
imageReader.onload = (event) => {
    imageLoaded = event.target.result;
    sendData();
};

const jsonReader = new FileReader();
jsonReader.onload = (event) => {
    jsonLoaded = event.target.result;
    sendData();
}

imageFileInput.onchange = ()=>{
    if (imageFileInput.files.length==0){
        imagePreviewContainer.style["display"] = "none";
        imageUpload.style["display"] = "inherit";
    }else{
        imagePreviewContainer.style["display"] = "flex";
        imageUpload.style["display"] = "none";
        previewImageFilename.textContent = imageFileInput.files[0].name
    }
}
annotation.onchange=()=>{
    if (annotation.files.length==0){
        annotationPreviewContainer.style["display"] = "none";
        annotationUpload.style["display"] = "inherit";
    }else{
        annotationPreviewContainer.style["display"] = "flex";
        annotationUpload.style["display"] = "none";
        previewAnnotationFilename.textContent = annotation.files[0].name
    }
}

function mouseMove(e, v) {
    floaty.style["display"] = "inherit"
    floaty.style["left"] = `${e.pageX}px`
    floaty.style["top"] = `${e.pageY}px`

   if (v!=null){
       lithology.textContent = v["lithology"]
       topDepth.textContent = v["top_depth"]
       depth.textContent = v["bottom_depth"] - v["top_depth"]
       topY.textContent = v["top_y"]
       yHeight.textContent = v["bottom_y"] - v["top_y"]
   }
}

function drawBox(result) {
    for (i of document.getElementsByClassName("box")) {
        i.remove()
    }

    console.log(result)
    const startY = result["start_y"]
    const realHeight = result["real_height"]
    const renderedHeight = realHeight

    for (let index = 0; index < result["result"].length; index++) {
        const i = result["result"][index]
        const box = document.createElement("div");
        box.classList.add("box")
        box.style["top"] = `${i["top_y"] / realHeight * renderedHeight - startY}px`
        box.style["height"] = `${i["bottom_y"] / realHeight * renderedHeight - i["top_y"] / realHeight * renderedHeight}px`
        box.style["z-index"] = "1"
        imageContainer.appendChild(box)

        box.onmouseenter = (e)=>mouseMove(e, i)
        box.onmousemove = (e)=>mouseMove(e,i)
        box.onmouseleave = (e) => {
            floaty.style["display"] = "none"
        }
    }
}

function sendData() {
    if (imageLoaded==null||jsonLoaded==null){
        return
    }

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/thing");
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
            imageResult.src = "data:image/jpeg;base64, " + data["image"]
            legend.src = "data:image/jpeg;base64, " + data["legend_image"]
            legend.style["display"]="inherit"
            legendEmpty.style["display"] = "none"
            drawBox(data)
        }
    }

    imageLoaded = null;
    jsonLoaded = null;

    button.disabled = false
}

button.onclick = () => {
    button.disabled = true

    imageReader.readAsDataURL(imageFileInput.files[0])
    jsonReader.readAsText(annotation.files[0])
}
