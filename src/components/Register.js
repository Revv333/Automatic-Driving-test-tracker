import React, { useState } from 'react';
import axios from 'axios'; // Make sure axios is installed: npm install axios
import { useNavigate } from 'react-router-dom';



function Register() {
  const [formValues, setFormValues] = useState({
    Email: '',
    Password: '',
    DateOfBirth: '',
    AadharNumber: '',
    Photo_name:'',
    Photo: null,
  });

  const [aadharError, setAadharError] = useState('');
  const [fileError, setFileError] = useState('');
  const [message, setMessage] = useState('');
  const navigate =useNavigate();

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormValues({
      ...formValues,
      [name]: value,
    });

    if (name === 'AadharNumber') {
      setAadharError(value.length !== 12 ? 'Enter Valid Aadhar Number' : '');
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && !selectedFile.name.match(/\.(jpg|jpeg)$/i)) {
      setFileError('Please upload a .jpeg file');
    } else {
      setFileError('');
      setFormValues({
        ...formValues,
        Photo_name:selectedFile.name,
        Photo: selectedFile,
      });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    
    const formData = new FormData();
    Object.entries(formValues).forEach(([key, value]) => {
      formData.append(key, value);
    });

    try {
      const response = await axios.post('http://127.0.0.1:5000/submit-form', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      console.log('Data sent:', response.data);
      setMessage('Registration successful');
      navigate('/');
      // Handle success, show a success message, etc.
    } catch (error) {
      console.error('Error sending data:', error);
      setMessage('Already Registered');
      // Handle error, show an error message, etc.
    }
  };

  return (
    <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
      <div className='card' style={{ width: '25rem' }}>
        <div className='card-body'>
          {message && <p>{message}</p>}
          <form onSubmit={handleSubmit}>
            <div className="mb-3">
              <label htmlFor="Email" className="form-label">Email address</label>
              <input type="email" className="form-control" id="Email" aria-describedby="emailHelp" name='Email' value={formValues.Email} onChange={handleInputChange}/>
              <div id="emailHelp" className="form-text">We'll never share your email with anyone else.</div>
            </div>
            <div className="mb-3">
              <label htmlFor="Password" className="form-label">Password</label>
              <input type="password" className="form-control" id="Password" name='Password' value={formValues.Password} onChange={handleInputChange}/>
            </div>
            <div className="mb-3">
              <label htmlFor="date" className="form-label">Date Of Birth</label>
              <input type="date" className="form-control" id="date" name='DateOfBirth' value={formValues.DateOfBirth} onChange={handleInputChange}/>
            </div>
            <div className="mb-3">
              <label htmlFor="aadharInput" className="form-label">Aadhar Number</label>
              <input
                type="text"
                className="form-control"
                id="aadharInput"
                maxLength={12}
                name='AadharNumber'
                value={formValues.AadharNumber}
                onChange={handleInputChange}
                required
              />
              {aadharError && <small style={{ color: 'red', font: "inherit" }}>{aadharError}</small>}
            </div>
            <div className="mb-3">
              <label htmlFor="photoUpload" className="form-label">Upload Your Photo</label>
              <input
                type="file"
                className="form-control"
                id="photoUpload"
                accept=".jpeg"
                onChange={handleFileChange}
              />
              {fileError && <small style={{ color: 'red' }}>{fileError}</small>}
            </div>

            <button type="submit" className="btn btn-primary">Submit</button>
          </form>
        </div>
      </div>
    </div>
  );
}

export default Register;
