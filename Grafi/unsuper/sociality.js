function load_sociality() {
        Bokeh.safely(function() {
          (function(root) {
            function embed_document(root) {
var x = document.getElementsByClassName("main-root")[0];
x.setAttribute('id','0002ce1a-2c47-4ad4-86ea-61e0a124b85b');
x.setAttribute('data-root-id', '11667');
render_items = [{"docid":"9dbfe4c6-99e9-48ac-b5a7-ed1b86a35a23","root_ids":["11667"],"roots":{"11667":"0002ce1a-2c47-4ad4-86ea-61e0a124b85b"}}];
root.Bokeh.embed.embed_items(docs_json, render_items);

            }
            if (root.Bokeh !== undefined) {
              embed_document(root);
            } else {
              var attempts = 0;
              var timer = setInterval(function(root) {
                if (root.Bokeh !== undefined) {
                  clearInterval(timer);
                  embed_document(root);
                } else {
                  attempts++;
                  if (attempts > 100) {
                    clearInterval(timer);
                    console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                  }
                }
              }, 10, root)
            }
          })(window);
        });
    };