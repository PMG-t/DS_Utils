function load_tourism() {
        Bokeh.safely(function() {
          (function(root) {
            function embed_document(root) {
var x = document.getElementsByClassName("main-root")[0];
x.setAttribute('id','c11074a0-b083-4bf3-9e57-6cdbf7ba1813');
x.setAttribute('data-root-id', '17130');
render_items = [{"docid":"a74338c6-828e-4682-9435-a29558e40f25","root_ids":["17130"],"roots":{"17130":"c11074a0-b083-4bf3-9e57-6cdbf7ba1813"}}];
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