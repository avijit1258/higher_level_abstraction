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

// use a V figure instead of MinusLine in the TreeExpanderButton
go.Shape.defineFigureGenerator("ExpandedLine", function(shape, w, h) {
  return new go.Geometry()
        .add(new go.PathFigure(0, 0.25*h, false)
              .add(new go.PathSegment(go.PathSegment.Line, .5 * w, 0.75*h))
              .add(new go.PathSegment(go.PathSegment.Line, w, 0.25*h)));
});

// use a sideways V figure instead of PlusLine in the TreeExpanderButton
go.Shape.defineFigureGenerator("CollapsedLine", function(shape, w, h) {
  return new go.Geometry()
        .add(new go.PathFigure(0.25*w, 0, false)
              .add(new go.PathSegment(go.PathSegment.Line, 0.75*w, .5 * h))
              .add(new go.PathSegment(go.PathSegment.Line, 0.25*w, h)));
});



function init() {
  
  var $ = go.GraphObject.make;  // for conciseness in defining templates

  myDiagram =
    $(go.Diagram, "fullDiagram",
      {
        allowMove: false,
        allowCopy: false,
        allowDelete: false,
        allowHorizontalScroll: false,
        layout:
          $(go.TreeLayout,
            {
              alignment: go.TreeLayout.AlignmentStart,
              angle: 0,
              compaction: go.TreeLayout.CompactionNone,
              layerSpacing: 16,
              layerSpacingParentOverlap: 1,
              nodeIndentPastParent: 1.0,
              nodeSpacing: 0,
              setsPortSpot: false,
              setsChildPortSpot: false
            })
      });

  myDiagram.nodeTemplate =
    $(go.Node,
      { // no Adornment: instead change panel background color by binding to Node.isSelected
        selectionAdorned: false,
        isTreeExpanded: false,
        // a custom function to allow expanding/collapsing on double-click
        // this uses similar logic to a TreeExpanderButton
        doubleClick: function(e, node) {
          var cmd = myDiagram.commandHandler;
          if (node.isTreeExpanded) {
            if (!cmd.canCollapseTree(node)) return;
          } else {
            if (!cmd.canExpandTree(node)) return;
          }
          e.handled = true;
          if (node.isTreeExpanded) {
            cmd.collapseTree(node);
          } else {
            cmd.expandTree(node);
          }
        }
      },
      $("TreeExpanderButton",
        { // customize the button's appearance
          "_treeExpandedFigure": "ExpandedLine",
          "_treeCollapsedFigure": "CollapsedLine",
          "ButtonBorder.fill": "whitesmoke",
          "ButtonBorder.stroke": null,
          "_buttonFillOver": "rgba(0,128,255,0.25)",
          "_buttonStrokeOver": null
        }),
      $(go.Panel, "Horizontal",
        { position: new go.Point(18, 0) },
        new go.Binding("background", "isSelected", function(s) { return (s ? "lightblue" : "white"); }).ofObject(),
        $(go.Picture,
          {
            width: 18, height: 18,
            margin: new go.Margin(0, 4, 0, 0),
            imageStretch: go.GraphObject.Uniform
          },
          // bind the picture source on two properties of the Node
          // to display open folder, closed folder, or document
          new go.Binding("source", "isTreeExpanded", imageConverter).ofObject(),
          new go.Binding("source", "isTreeLeaf", imageConverter).ofObject()),
        $(go.TextBlock,
          { font: '9pt Verdana, sans-serif' },
          new go.Binding("text", "node_text", function(s) { return " " + s; }))
      )  // end Horizontal Panel
    );  // end Node

  // without lines
  myDiagram.linkTemplate = $(go.Link);


  myDiagram.addDiagramListener("ObjectContextClicked",
    function (e) {
      var part = e.subject.part;
      if (!(part instanceof go.Link)) {
        // showUserStudyPanel(part);
        showNodeDetails(part);
      }

    });

  setupDiagram();

}

function imageConverter(prop, picture) {
  var node = picture.part;
  if (node.isTreeLeaf) {
    return "/static/images/document.svg";
  } else {
    if (node.isTreeExpanded) {
      return "/static/images/openFolder.svg";
    } else {
      return "/static/images/closedFolder.svg";
    }
  }
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


function clearDiagram() {
  myDiagram.model = null;
}




