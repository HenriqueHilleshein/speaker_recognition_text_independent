function getJSONFile(filename, callback) {
    var xmlhttp = new XMLHttpRequest();
    var url = filename;

    xmlhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
              callback(JSON.parse(this.responseText))
        }
    };
    xmlhttp.open("GET", url, false);
    xmlhttp.send();
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

var audio_context
var recorder
async function startRecording(button) {
    await sleep(300)
    recorder && recorder.record();
    button.disabled = true;
    button.nextElementSibling.disabled = false;
}

function stopRecording(button) {
    recorder && recorder.stop();
    button.disabled = true;
    button.previousElementSibling.disabled = false;
    // create WAV download link using audio data blob
    createDownloadLink(button.id);
    recorder.clear();
}
  function createDownloadLink(id) {
    recorder && recorder.exportWAV(function(blob) {
      var url = URL.createObjectURL(blob);
      var li = document.createElement('li');
      var au = document.createElement('audio');
      var hf = document.createElement('a');
      
      au.controls = true;
      au.src = url;
      hf.href = url;
      hf.download = new Date().toISOString() + '.wav';
      hf.innerHTML = hf.download;
      li.appendChild(au);
      li.appendChild(hf);
      if(id != ""){
          var recordingslist = document.getElementById("recordingslist" + id)
          recordingslist.appendChild(li);
          var progressBar = document.getElementById('progress' + id);
          var filename = './trainwav_' + id + '.wav'
      } else {
          var recordingslist = document.getElementById("recordingslist" + id)
          recordingslist.appendChild(li);
          var progressBar = document.getElementById('progress');
	  var filename = "mywav.wav" 
      }
      upload(blob, filename, progressBar)
    });     
  }

function get_microphone() {
    // Older browsers might not implement mediaDevices at all, so we set an empty object first
    if (navigator.mediaDevices === undefined) {
        navigator.mediaDevices = {};
    }
    // Some browsers partially implement mediaDevices. We can't just assign an object
    // with getUserMedia as it would overwrite existing properties.
    // Here, we will just add the getUserMedia property if it's missing.
    if (navigator.mediaDevices.getUserMedia === undefined) {
        navigator.mediaDevices.getUserMedia = function(constraints) {

            // First get ahold of the legacy getUserMedia, if present
            var getUserMedia = navigator.webkitGetUserMedia || navigator.getUserMedia;

            // Some browsers just don't implement it - return a rejected promise with an error
            // to keep a consistent interface
            if (!getUserMedia) {
                return Promise.reject(new Error('getUserMedia is not implemented in this browser'));
            }

            // Otherwise, wrap the call to the old navigator.getUserMedia with a Promise
            return new Promise(function(resolve, reject) {
            getUserMedia.call(navigator, constraints, resolve, reject);
            });
        }
    } 

    navigator.mediaDevices.getUserMedia({ audio: true, video: false })
    .then(function(stream){
        var input = audio_context.createMediaStreamSource(stream);
        // Uncomment if you want the audio to feedback directly
        // input.connect(audio_context.destination);
        // __log('Input connected to audio context destination.');
    
        recorder = new Recorder(input);
    })
}
window.onload = function init() {
    try {
       // webkit shim
       window.AudioContext = window.AudioContext || window.webkitAudioContext;
       window.URL = window.URL || window.webkitURL;
       audio_context = new AudioContext;
       get_microphone()
    } catch (e) {
       alert('No web audio support in this browser!');
    }
};
