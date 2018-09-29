

export const registerUser = (user) => {

    return fetch('/auth/register', {
        method: 'post',
        body: JSON.stringify(user),
        headers: new Headers({
            'content-type': 'application/json'
        })
    }).then(
        response => {
            try{
                if(response.ok) {
                    return response.json();
                }
                return response.json().then(results => Promise.regect(results));
                
            }
            catch(error) {
                return response.text().then(error => Promise.reject(error));
            }
        }
    )
}