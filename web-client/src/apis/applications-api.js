

export const parseCertificate = (file) => {

    let formData = new FormData();
    formData.append('file', file)
    return fetch('/auth/parse-certificate',
        {method: 'POST', 
         body: formData
        }
    ).then( response => {
        if(response.ok){
            return response.json();
        }

        try{
            // in case is error we know about
            return response.json();
        }
        catch(error) {
            return response.text().then(error => Promise.reject(error));
        } 
    })
}

export const assessCourseOrCert = (payload) => {


    return fetch('/auth/assessment',
            {method: 'POST', body: JSON.stringify(payload),
            headers: new Headers({
                'content-type': 'application/json'
            })
            })
            .then( response => {
                if (response.ok) {
                    return response.json();
                }

                try{
                    return response.json();
                }
                catch(error) {
                    return response.text().then(error => Promise.reject(error));
                }
            })
}