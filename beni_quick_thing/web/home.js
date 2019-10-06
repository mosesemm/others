
$(document).ready(function(){

    function Home() {

        var that = this;
        that.video = document.getElementById("play_area");
        that.videoId =  ko.observable();


        that.selectVideo = function(data, event) {
            that.video.setAttribute("src", this.video)
            that.videoId(this.video);
            that.video.load();
        }
        that.play =  function() {
            that.video.play();

            $.ajax({
                type: "POST",
                url: "api/play/"+that.videoId(),
                contentType: "application/json",
                data: JSON.stringify({}),
                dataType: "json",
                success: function(data){
                    alert(data.message);
                },
                error: function(){
                    alert("some error happened");
                }
            });
        }
        that.survey = function() {
            window.location = "survey.html";
        }
    }

    ko.applyBindings(new Home());
});