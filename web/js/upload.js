function upload(blobOrFile, filename, progressBar) {
  var xhr = new XMLHttpRequest();
  xhr.open('POST', filename, true);
  xhr.onload = function(e) {};
  xhr.upload.onprogress = function(e) {
    if (e.lengthComputable) {
      progressBar.value = (e.loaded / e.total) * 100;
      progressBar.textContent = progressBar.value; // Fallback for unsupported browsers.
    }
  };
 
  xhr.send(blobOrFile);
}

function postJSONFile(filename, data, callback) {
    var xhr = new XMLHttpRequest();
    xhr.open("POST", filename, false);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4) {
            if(xhr.status === 200){
                callback(xhr.response);
            }
        }

    };
    xhr.send(JSON.stringify(data));
}
