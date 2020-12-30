import "./App.css";
import React, { useState } from "react";
import axios from "axios";

function App() {
  const [image, setImage] = useState("");
  const [caption, setCaption] = useState("");

  const handelInput = (file) => {
    let reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = function (e) {
      let dataURL = reader.result;
      console.log(dataURL);
      setImage(dataURL);
    };
  };

  const upload = () => {
    axios
      .post(
        "http://127.0.0.1:5000/predict",
        { image: image.replace("data:image/png;base64,", "") },
        {
          onUploadProgress: (ProgressEvent) => {
            console.log(
              "upload:" +
                Math.round((ProgressEvent.loaded / ProgressEvent.total) * 100) +
                "%"
            );
          },
        }
      )
      .then((res) => setCaption(res.data.prediction));
  };

  return (
    <div className="App">
      <div className="header">
        <span>Neural Image Captioning</span>
      </div>
      <div className="main">
        <input
          className="upload"
          type="file"
          accept="image/*"
          onChange={(e) => handelInput(e.target.files[0])}
        />
        <button onClick={upload}>Generate</button>
        <img className="image" src={image} alt="" height="224" width="224" />
        <p>Predicted Caption:{caption}</p>
      </div>
    </div>
  );
}

export default App;
