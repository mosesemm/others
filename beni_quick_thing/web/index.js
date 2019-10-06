
$(document).ready(function(){

    function Login() {

        var that = this;

        that.username = ko.observable();
        that.password = ko.observable();

        that.getData = function(){
            return {
                username: that.username(),
                password: that.password()
            }
        }

        that.login = function() {

            $.ajax({
                type: "POST",
                url: "api/login",
                contentType: "application/json",
                data: JSON.stringify(that.getData()),
                dataType: "json",
                success: function(data){
                    window.location = "home.html";
                },
                error: function(){
                    alert("some error happened");
                }
            });
        }
    }

    ko.applyBindings(new Login());
});