
$(document).ready(function(){

    function Survey() {

        var that = this;


        that.results = function() {
            window.location = "results.html";
        }
    }

    ko.applyBindings(new Survey());
});