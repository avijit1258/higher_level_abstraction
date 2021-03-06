function get_cluster() {
    Url = 'http://127.0.0.1:5000/get_cluster'
    var subject_system = document.getElementById('subject_system_id').value;
  
  
    $.getJSON(Url, {
      subject_system: subject_system
    }, function (result) {
      // alert(result);
  
      setupDiagram(result);
  
      console.log(result);
    })
    clearDiagram();
  
  }

  function setupDiagram(result) {
    var nodeDataArray = [];
    for (x in result) {
      nodeDataArray.push({
        key: result[x].key,
        parent: result[x].parent,
        node_text: result[x].tfidf_word,
        tfidf_word: result[x].tfidf_word,
        tfidf_method: result[x].tfidf_method,
        lda_word: result[x].lda_word,
        lda_method: result[x].lda_method,
        lsi_word: result[x].lsi_word,
        lsi_method: result[x].lsi_method,
        color: "#cce6ff",
        spm_method: result[x].spm_method,
        text_summary: result[x].text_summary,
        files: result[x].files,
        files_count: result[x].files_count
      });
  
    }
    // Use below line for randomly coloring brushes
    // color: go.Brush.randomColor()
  
    myDiagram.model = new go.TreeModel(nodeDataArray);
    // update_nodes_for_study();
  
  }
  
  function update_node_text(node, technique) {
    myDiagram.model.commit(function (m) { // this Model
      // This is the safe way to change model data
      // GoJS will be notified that the data has changed
      // and can update the node in the Diagram
      // and record the change in the UndoManager
      if (technique == 'tfidf_word') {
        m.set(node.data, "node_text", node.data.tfidf_word);
      } else if (technique == 'tfidf_method') {
        m.set(node.data, "node_text", node.data.tfidf_method);
      } else if (technique == 'lda_word') {
        m.set(node.data, "node_text", node.data.lda_word);
      } else if (technique == 'lda_method') {
        m.set(node.data, "node_text", node.data.lda_method);
      } else if (technique == 'lsi_word') {
        m.set(node.data, "node_text", node.data.lsi_word);
      } else if (technique == 'lsi_method') {
        m.set(node.data, "node_text", node.data.lsi_method);
      }
  
    }, "update node text");
  }
  
  
  function showNodeDetails(part){
    var clickable_text = '';
  
    for(index = 0; index < part.data.files.length; index++){
      clickable_text += '<a class="click_text" >' + part.data.files[index] + '</a>, ';
    }
  
    document.getElementById('node_key').innerHTML = 'Node Key: ' + part.data.key;
    document.getElementById('node_summary').innerHTML = part.data.text_summary;
    document.getElementById('node_patterns').innerHTML = part.data.spm_method;
    document.getElementById('files').innerHTML = clickable_text;
    document.getElementById('number_of_files').innerHTML = part.data.files_count;
  
    // jQuery('#node_details').modal('show');
  }
  
  
  function change_node_color(){
    myDiagram.nodes.each(function (n) {
      myDiagram.model.commit(function (m) {
        m.set(n.data, "color", "#ffc61a" );
      }, 'change node color');
    });
  }
  
  jQuery('#search_button').click(function(){
    alert('hello');
    change_node_color();
  });
  
  jQuery('#technique_choice_id').change(function () {
    var technique_choice = document.getElementById('technique_choice_id').value;
  
    myDiagram.nodes.each(function (n) {
      update_node_text(n, technique_choice);
    });
  
  });
  