function load_history() {
        Bokeh.safely(function() {
          (function(root) {
            function embed_document(root) {
var x = document.getElementsByClassName("main-root")[0];
x.setAttribute('id','49990761-8996-439f-938e-6f1da4ce4ad5');
x.setAttribute('data-root-id', '14297');
render_items = [{"docid":"d6e9ed8f-0552-4bc0-a537-6d6f74a0320f","root_ids":["14297"],"roots":{"14297":"49990761-8996-439f-938e-6f1da4ce4ad5"}}];
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