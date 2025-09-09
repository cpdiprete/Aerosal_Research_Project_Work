// import React, { useState } from 'react';

// function App() {
// const [result, setResult] = useState(null);

// const onFileChange = async e => {
//     const file = e.target.files[0];
//     const form = new FormData();
//     form.append('file', file);
//     form.append('x', 100);
//     form.append('y', 100);
//     form.append('box_size', 50);

//     const resp = await fetch('http://localhost:8000/process-video', {
//     method: 'POST',
//     body: form
//     });
//     const json = await resp.json();
//     setResult(json);
// };

// return (
//     <div style={{ padding: 20 }}>
//     <h1>Upload your video</h1>
//     <input type="file" accept="video/*" onChange={onFileChange}/>
//     {result && (
//         <div>
//         <p>Detected {result.count} droplets.</p>
//         <pre>{JSON.stringify(result, null, 2)}</pre>
//         </div>
//     )}
//     </div>
// );
// }

// export default App;
import { useState } from "react";
import axios from "axios";

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string>("");
  const [status, setStatus] = useState<string>("");

  const handleUpload = async () => {
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    const { data } = await axios.post(
      "http://localhost:8000/process-video",
      form,
      { headers: { "Content-Type": "multipart/form-data" } }
    );
    setJobId(data.job_id);
    pollStatus(data.job_id);
  };

  const pollStatus = async (id: string) => {
    const poll = setInterval(async () => {
      const { data } = await axios.get(`http://localhost:8000/status/${id}`);
      setStatus(data.status);
      if (data.status === "done") {
        clearInterval(poll);
        window.location.href = `http://localhost:8000/result/${id}`; // or fetch/display inline
      }
      if (data.status === "error") clearInterval(poll);
    }, 2000);
  };

  return (
    <div style={{ padding: 40 }}>
      <h1>Video Analyzer</h1>
      <input
        type="file"
        accept="video/*"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
      />
      <button onClick={handleUpload} disabled={!file}>
        Upload & Run
      </button>
      {status && <p>Job status: {status}</p>}
    </div>
  );
}
