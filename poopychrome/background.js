chrome.contextMenus.create({
    title : "How poopy is this?",
    contexts: ['image'],
    onclick : function(info, tab) {
        jQuery.getJSON(
            "http://earlspeaks.ngrok.com/api/image_to_captions?url=" + info.srcUrl,
            function(data) {
                var captions = "";
                for (var img=0; img < data.data.images.length; img++) {
                    var info = data.data.images[img].captions;
                    for (var i=0; i<info.length; i++) {
                        captions += img + ") " + info[i][1] + " [" + info[i][0] + "]\n";
                    }
                }
                alert(captions);
            }
        ).fail(function() {
            alert("Could not process image");
        });
    }
});
